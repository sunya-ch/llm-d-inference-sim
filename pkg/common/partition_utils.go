/*
Copyright 2025 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package common

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
)

// kvCacheBytesPerElement returns the number of bytes for one KV cache element.
// Unknown or empty dtype falls back to float16 (2 bytes).
func kvCacheBytesPerElement(dtype string) int {
	switch strings.ToLower(dtype) {
	case "float8", "fp8":
		return 1
	default: // float16, bfloat16, auto, empty — all 2 bytes
		return 2
	}
}

const hfTokenEnv = "HF_TOKEN"

var hfConfigHTTPClient = &http.Client{Timeout: 30 * time.Second}

type hfModelConfig struct {
	NumHiddenLayers   int `json:"num_hidden_layers"`
	NumKVHeads        int `json:"num_key_value_heads"`
	HeadDim           int `json:"head_dim"`
	HiddenSize        int `json:"hidden_size"`
	NumAttentionHeads int `json:"num_attention_heads"`
}

func derivedHeadDim(headDim, hiddenSize, numAttentionHeads int) int {
	if headDim > 0 {
		return headDim
	}
	if hiddenSize <= 0 || numAttentionHeads <= 0 || hiddenSize%numAttentionHeads != 0 {
		return 0
	}
	return hiddenSize / numAttentionHeads
}

func lookupHFModelConfig(model string) (*hfModelConfig, error) {
	if model == "" {
		return nil, fmt.Errorf("model is empty")
	}

	req, err := http.NewRequest(http.MethodGet, "https://huggingface.co/"+model+"/resolve/main/config.json", nil)
	if err != nil {
		return nil, err
	}
	if token := os.Getenv(hfTokenEnv); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := hfConfigHTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface config request failed with status %d", resp.StatusCode)
	}

	var cfg hfModelConfig
	if err := json.NewDecoder(resp.Body).Decode(&cfg); err != nil {
		return nil, err
	}
	cfg.HeadDim = derivedHeadDim(cfg.HeadDim, cfg.HiddenSize, cfg.NumAttentionHeads)
	if cfg.NumHiddenLayers <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("huggingface config missing required fields")
	}
	return &cfg, nil
}

// kvCacheDerivationResult holds the outcome and all computed intermediates from
// deriveKVCacheBlocks so that callers (e.g. Show) can log them without re-running
// the derivation.
type kvCacheDerivationResult struct {
	skipped       bool
	skipReason    string
	blocks        int
	gpuBytes      int64
	bpe           int
	bytesPerBlock int
	availableBytes float64
}

// deriveKVCacheBlocks computes the number of KV cache blocks that fit in available
// GPU memory, mirroring vLLM's startup formula:
//
//	available  = gpuMemoryBytes × gpuMemUtilization − modelWeightsBytes
//	bytesPerBlock = 2 × numLayers × numKVHeads × headDim × bytesPerElem × blockSize
//	blocks     = floor(available / bytesPerBlock)
//
// Returns (0, false) if any input is non-positive or available memory ≤ 0.
func deriveKVCacheBlocks(c *Configuration) (int, bool) {
	res, _ := deriveKVCacheBlocksWithResult(c)
	return res.blocks, !res.skipped
}

// deriveKVCacheBlocksWithResult is like deriveKVCacheBlocks but also returns a
// kvCacheDerivationResult populated with intermediate values for logging.
func deriveKVCacheBlocksWithResult(c *Configuration) (kvCacheDerivationResult, bool) {
	skip := func(reason string) (kvCacheDerivationResult, bool) {
		return kvCacheDerivationResult{skipped: true, skipReason: reason}, false
	}

	// Require the GPU memory env var to be set.
	raw := os.Getenv(c.GPUMemoryEnvVar)
	if raw == "" {
		return skip("GPU memory env var not set")
	}

	// Parse Kubernetes quantity string (e.g. "20Gi", "21474836480").
	q, err := resource.ParseQuantity(raw)
	if err != nil {
		return skip("failed to parse GPU memory value")
	}
	gpuBytes, ok := q.AsInt64()
	if !ok || gpuBytes <= 0 {
		return skip("GPU memory value out of range")
	}

	headDim := c.HeadDim

	// All four arch params must be provided.
	if c.NumHiddenLayers <= 0 || c.NumKVHeads <= 0 || headDim <= 0 || c.ModelWeightsSizeGB <= 0 {
		return skip("missing architecture parameters")
	}

	bpe := kvCacheBytesPerElement(c.KVCacheDtype)
	bytesPerBlock := 2 * c.NumHiddenLayers * c.NumKVHeads * headDim * bpe * c.TokenBlockSize
	if bytesPerBlock <= 0 {
		return skip("computed bytes-per-block is zero")
	}

	available := float64(gpuBytes)*c.GPUMemoryUtilization - c.ModelWeightsSizeGB*float64(1<<30)
	if available <= 0 {
		return skip("no memory available after subtracting model weights")
	}

	blocks := int(math.Floor(available / float64(bytesPerBlock)))
	if blocks <= 0 {
		return skip("computed block count is zero")
	}

	return kvCacheDerivationResult{
		blocks:        blocks,
		gpuBytes:      gpuBytes,
		bpe:           bpe,
		bytesPerBlock: bytesPerBlock,
		availableBytes: available,
	}, true
}

// deriveSMFactor reads the GPU compute percentage from c.GPUComputeEnvVar and
// returns (100/compute, true) as the static SM latency multiplier to apply to
// all GPU-bound latency parameters at startup.
// Returns (0, false) if the variable is unset, unparseable, or out of range [1, 100].
func deriveSMFactor(c *Configuration) (float64, bool) {
	raw := os.Getenv(c.GPUComputeEnvVar)
	if raw == "" {
		return 0, false
	}
	p, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || p < 1 || p > 100 {
		return 0, false
	}
	return 100.0 / float64(p), true
}
