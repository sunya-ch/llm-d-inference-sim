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
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// setEnv sets an env var for the duration of a single spec and restores it after.
func setEnv(key, value string) {
	old, hadOld := os.LookupEnv(key)
	if value == "" {
		os.Unsetenv(key) //nolint:errcheck
	} else {
		os.Setenv(key, value) //nolint:errcheck
	}
	DeferCleanup(func() {
		if hadOld {
			os.Setenv(key, old) //nolint:errcheck
		} else {
			os.Unsetenv(key) //nolint:errcheck
		}
	})
}

// baseKVConfig returns a config matching the doc example:
// 32 layers, 8 KV heads, head_dim 128, 16 GiB weights, 20 GiB GPU, util 0.9,
// block_size 16, float16 → 128 blocks.
func baseKVConfig() *Configuration {
	return &Configuration{
		GPUMemoryEnvVar:      "TEST_GPU_MEM",
		GPUMemoryUtilization: 0.9,
		ModelWeightsSizeGB:   16,
		NumHiddenLayers:      32,
		NumKVHeads:           8,
		HeadDim:              128,
		KVCacheDtype:         "float16",
		TokenBlockSize:       16,
	}
}

var _ = Describe("derivedHeadDim", func() {
	DescribeTable("returns explicit or derived head_dim",
		func(headDim, hiddenSize, numAttentionHeads, expected int) {
			Expect(derivedHeadDim(headDim, hiddenSize, numAttentionHeads)).To(Equal(expected))
		},
		Entry("uses explicit head_dim when present", 128, 4096, 32, 128),
		Entry("derives from hidden_size and num_attention_heads", 0, 4096, 32, 128),
		Entry("returns zero when hidden_size is not divisible", 0, 4097, 32, 0),
		Entry("returns zero when hidden_size is missing", 0, 0, 32, 0),
		Entry("returns zero when num_attention_heads is missing", 0, 4096, 0, 0),
	)
})

var _ = Describe("kvCacheBytesPerElement", func() {
	DescribeTable("maps dtype to bytes per element",
		func(dtype string, expected int) {
			Expect(kvCacheBytesPerElement(dtype)).To(Equal(expected))
		},
		Entry("float16", "float16", 2),
		Entry("bfloat16", "bfloat16", 2),
		Entry("auto", "auto", 2),
		Entry("empty string", "", 2),
		Entry("float8", "float8", 1),
		Entry("fp8 alias", "fp8", 1),
		Entry("uppercase FP8", "FP8", 1),
	)
})

var _ = Describe("deriveKVCacheBlocks", func() {
	DescribeTable("returns the correct block count",
		func(envValue string, mutate func(*Configuration), expectOK bool, expectBlocks int) {
			setEnv("TEST_GPU_MEM", envValue)
			c := baseKVConfig()
			if mutate != nil {
				mutate(c)
			}
			blocks, ok := deriveKVCacheBlocks(c)
			Expect(ok).To(Equal(expectOK))
			if expectOK {
				Expect(blocks).To(Equal(expectBlocks))
			}
		},
		// --- happy path ---
		// bytes_per_block = 2 × 32 × 8 × 128 × 2 × 16 = 2,097,152 (2 MiB)
		// available = 20 GiB × 0.9 − 16 GiB = 2 GiB = 2,147,483,648
		// blocks = floor(2,147,483,648 / 2,097,152) = 1024
		Entry("Llama-3.1-8B on 20 GiB → 1024 blocks",
			"20Gi", nil, true, 1024),
		Entry("decimal quantity string (20 GiB as bytes)",
			"21474836480", nil, true, 1024),
		// float8 halves bytes_per_element (1 instead of 2) → bytes_per_block = 1,048,576
		// blocks = floor(2,147,483,648 / 1,048,576) = 2048
		Entry("float8 dtype halves bytes_per_block → 2048 blocks",
			"20Gi", func(c *Configuration) { c.KVCacheDtype = "float8" }, true, 2048),

		// --- env var problems ---
		Entry("env var not set",
			"", nil, false, 0),
		Entry("env var not a valid quantity",
			"not-a-number", nil, false, 0),

		// --- missing arch params ---
		Entry("NumHiddenLayers is zero",
			"20Gi", func(c *Configuration) { c.NumHiddenLayers = 0 }, false, 0),
		Entry("NumKVHeads is zero",
			"20Gi", func(c *Configuration) { c.NumKVHeads = 0 }, false, 0),
		Entry("HeadDim is zero",
			"20Gi", func(c *Configuration) { c.HeadDim = 0 }, false, 0),
		Entry("ModelWeightsSizeGB is zero",
			"20Gi", func(c *Configuration) { c.ModelWeightsSizeGB = 0 }, false, 0),

		// --- memory exhausted ---
		// 20 GiB × 0.9 = 18 GiB available, weights = 20 GiB → available < 0
		Entry("weights exceed available memory",
			"20Gi", func(c *Configuration) { c.ModelWeightsSizeGB = 20 }, false, 0),
	)
})
