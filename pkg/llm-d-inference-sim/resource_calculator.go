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

package llmdinferencesim

import (
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	MaxMemoryOverheadFactor = 1.3 // Allow up to 30% overhead for framework variations
	MinMaxThreads           = 30720
	MinHiddenSize           = 128
)

// GPUSpecs contains specifications for different GPU types
type GPUSpecs struct {
	Name                           string
	VRAM                           int     // GB
	MemoryBW                       float64 // TB/s
	PeakTFLOPS                     float64 // FP16/BF16
	PeakFP8                        float64 // FP8 TFLOPS
	MaxThreads                     int     // Typical hardware thread capacity
	UnderprovisionContentionImpact float64 // 0.0 to 1.0
	OverprovisionSpeedUp           float64 // 0.0 to 1.0
}

var gpuRegistry = map[string]GPUSpecs{
	"NVIDIA-H200-SXM5-141GB": {Name: "NVIDIA-H200-SXM5-141GB", VRAM: 141, MemoryBW: 4.8, PeakTFLOPS: 989.0, PeakFP8: 3958.0, MaxThreads: 249856, UnderprovisionContentionImpact: 0.08, OverprovisionSpeedUp: 0.5},
	"NVIDIA-H100-SXM5-80GB":  {Name: "NVIDIA-H100-SXM5-80GB", VRAM: 80, MemoryBW: 3.35, PeakTFLOPS: 989.0, PeakFP8: 1979.0, MaxThreads: 184320, UnderprovisionContentionImpact: 0.15, OverprovisionSpeedUp: 0.35},
	"NVIDIA-A100-SXM4-80GB":  {Name: "NVIDIA-A100-SXM4-80GB", VRAM: 80, MemoryBW: 2.0, PeakTFLOPS: 312.0, PeakFP8: 0.0, MaxThreads: 108544, UnderprovisionContentionImpact: 0.3, OverprovisionSpeedUp: 0.2},
	"NVIDIA-L4-24GB":         {Name: "NVIDIA-L4-24GB", VRAM: 24, MemoryBW: 0.3, PeakTFLOPS: 121.0, PeakFP8: 242.0, MaxThreads: 30720, UnderprovisionContentionImpact: 0.7, OverprovisionSpeedUp: 0.05},
}

// ArchParams contains model architecture parameters
type ArchParams struct {
	ParametersB float64
	Layers      int
	KVHeads     int
	HiddenSize  int
	QueryHeads  int
	Precision   int
}

var modelFamilyMap = map[string]ArchParams{
	"llama-3-8b":   {ParametersB: 8.0, Layers: 32, KVHeads: 8, HiddenSize: 4096, QueryHeads: 32, Precision: 2},
	"llama-3.1-8b": {ParametersB: 8.0, Layers: 32, KVHeads: 8, HiddenSize: 4096, QueryHeads: 32, Precision: 2},
	"llama-3-70b":  {ParametersB: 70.0, Layers: 80, KVHeads: 8, HiddenSize: 8192, QueryHeads: 64, Precision: 2},
	"mistral-nemo": {ParametersB: 12.0, Layers: 40, KVHeads: 8, HiddenSize: 5120, QueryHeads: 32, Precision: 2},
	"phi-3-mini":   {ParametersB: 3.8, Layers: 32, KVHeads: 8, HiddenSize: 3072, QueryHeads: 32, Precision: 2},
	"phi-3-medium": {ParametersB: 14.0, Layers: 40, KVHeads: 8, HiddenSize: 5120, QueryHeads: 40, Precision: 2},
}

// ComplexityMetrics contains model complexity information
type ComplexityMetrics struct {
	FlopsPerToken       float64 // Total FLOPs for one forward pass of one token
	ArithmeticIntensity float64 // FLOPs / Bytes (Determines if Compute or Memory bound)
	MinThreads          int     // Suggested minimum threads to saturate the hidden dimension
}

// GPUResourceLimits holds the resource limits for a single GPU device
type GPUResourceLimits struct {
	// DeviceID is the GPU device identifier (e.g., 0, 1, 2, etc.)
	DeviceID int
	// ActiveThreadPercentage is the percentage of GPU threads that can be actively used (0-100)
	ActiveThreadPercentage float64
	// MemoryLimitBytes is the memory limit in bytes
	MemoryLimitBytes int64
}

// ResourceConsumption represents the estimated resource consumption for a request
type ResourceConsumption struct {
	// ActiveThreadPercentage is the estimated percentage of active GPU threads used (0-100)
	ActiveThreadPercentage float64
	// MemoryUsageBytes is the estimated memory usage in bytes
	MemoryUsageBytes int64
	// DeviceID is the GPU device this consumption applies to
	DeviceID int
}

// ResourceCalculator estimates GPU resource consumption based on input and model
type ResourceCalculator interface {
	// CalculateResourceConsumption estimates resource consumption for a request
	// Returns a slice of ResourceConsumption, one for each GPU device
	CalculateResourceConsumption(params *ResourceParams) []ResourceConsumption
	// GetGPULimits returns the configured GPU resource limits
	GetGPULimits() []GPUResourceLimits
}

// ResourceParams contains parameters for resource consumption calculation
type ResourceParams struct {
	// PromptTokens is the number of input tokens
	PromptTokens int
	// GenerationTokens is the number of tokens to generate
	GenerationTokens int
	// CachedPromptTokens is the number of cached prompt tokens
	CachedPromptTokens int
	// RunningReqs is the number of currently running requests
	RunningReqs int64
	// ModelName is the name of the model being used
	ModelName string
	// KVCacheUsagePercentage is the current KV cache utilization (0-1)
	KVCacheUsagePercentage float64
	// KVCacheSizeTokens is the total KV cache capacity in tokens
	KVCacheSizeTokens int
}

// queueTheoryResourceCalculator implements ResourceCalculator using queue theory
type queueTheoryResourceCalculator struct {
	gpuLimits                   []GPUResourceLimits
	modelID                     string
	validParamsInBillions       bool
	paramsFound                 bool
	params                      *ArchParams
	complexity                  ComplexityMetrics
	effectiveBatchSize          float64
	maxNumBatchedTokens         *int32
	modelSize                   float64
	memoryPerToken              float64
	activeMemoryPerPrefillToken float64
	gpuFound                    bool
	gpuSpec                     *GPUSpecs
	memoryOverheadFactor        float64
	minMemoryUsage              float64
	logger                      logr.Logger
}

// NewResourceCalculator creates a new resource calculator based on configuration
func NewResourceCalculator(config *common.Configuration) (ResourceCalculator, error) {
	gpuLimits, err := parseGPULimitsFromEnv()
	if err != nil {
		return nil, fmt.Errorf("failed to parse GPU limits from environment: %w", err)
	}

	// If no GPU limits found in environment, return nil (resource tracking disabled)
	if len(gpuLimits) == 0 {
		return nil, nil
	}

	// Create logger
	logger := logr.Discard() // Use discard logger by default

	// Determine effective batch size
	effectiveBatchSize := float64(config.MaxNumSeqs)

	// Get GPU name from environment or use default
	gpuName := os.Getenv("GPU_MODEL_NAME_0")
	if gpuName == "" {
		gpuName = "NVIDIA-A100-SXM4-80GB" // default
	}

	// Infer model parameters
	paramsFound, params := inferredArchParams(config.Model, logger)
	validParamsInBillions, _ := extractParams(config.Model, logger)

	// Get GPU specs
	gpuFound, gpuSpec := getGPUSpecs(gpuName, logger)
	if !gpuFound {
		_, gpuSpec = getGPUSpecs("NVIDIA-A100-SXM4-80GB", logger)
	}

	return &queueTheoryResourceCalculator{
		gpuLimits:                   gpuLimits,
		modelID:                     config.Model,
		validParamsInBillions:       validParamsInBillions,
		paramsFound:                 paramsFound,
		params:                      &params,
		complexity:                  estimateModelComplexity(config.Model, params.ParametersB, params.HiddenSize),
		effectiveBatchSize:          effectiveBatchSize,
		maxNumBatchedTokens:         nil,
		modelSize:                   estimateModelSizeQT(config.Model, params.ParametersB, params.Precision),
		memoryPerToken:              estimateMemoryPerTokenFromID(params, params.Precision),
		activeMemoryPerPrefillToken: estimateActiveMemoryPerTokenFromID(effectiveBatchSize, params, params.Precision),
		gpuFound:                    gpuFound,
		gpuSpec:                     gpuSpec,
		memoryOverheadFactor:        1.2, // default 20% - more realistic for vLLM
		logger:                      logger,
	}, nil
}

// CalculateResourceConsumption estimates resource consumption using queue theory
func (r *queueTheoryResourceCalculator) CalculateResourceConsumption(params *ResourceParams) []ResourceConsumption {
	if len(r.gpuLimits) == 0 {
		return []ResourceConsumption{}
	}

	numGPUs := len(r.gpuLimits)

	// Calculate cached hit ratio
	cachedHitRatio := 0.0
	if params.PromptTokens > 0 {
		cachedHitRatio = float64(params.CachedPromptTokens) / float64(params.PromptTokens)
	}

	// Calculate prefill tokens (tokens that actually need to be computed)
	prefillTokens := float64(params.PromptTokens) * (1.0 - cachedHitRatio) * r.effectiveBatchSize
	if r.maxNumBatchedTokens != nil {
		prefillTokens = math.Min(prefillTokens, float64(*r.maxNumBatchedTokens))
	}

	// Estimate thread occupancy
	threadOccupancy := r.estimateThreadOccupancy(r.effectiveBatchSize, prefillTokens, numGPUs)

	// Estimate memory usage
	memoryUsageGB := r.estimateMemoryUsage(params.PromptTokens, params.GenerationTokens, prefillTokens, numGPUs)

	// Convert to per-GPU values
	threadPerGPU := threadOccupancy / float64(numGPUs)
	memoryPerGPU := int64(memoryUsageGB * 1e9 / float64(numGPUs))

	// Create consumption entries for each GPU
	consumptions := make([]ResourceConsumption, numGPUs)
	for i, gpuLimit := range r.gpuLimits {
		threadUtilization := threadPerGPU
		memoryUsage := memoryPerGPU

		// Apply memory limit constraint
		if memoryUsage > gpuLimit.MemoryLimitBytes {
			log.Printf("[ResourceCalculator] Memory usage %d bytes (%.2f GB) per GPU exceeds limit %d bytes (%.2f GB) (DeviceID: %d)",
				memoryUsage, float64(memoryUsage)/(1024*1024*1024),
				gpuLimit.MemoryLimitBytes, float64(gpuLimit.MemoryLimitBytes)/(1024*1024*1024),
				gpuLimit.DeviceID)
			memoryUsage = gpuLimit.MemoryLimitBytes
		}

		// Apply thread percentage limit constraint
		if threadUtilization > gpuLimit.ActiveThreadPercentage {
			log.Printf("[ResourceCalculator] Thread utilization %.2f%% per GPU exceeds limit %.2f%% (DeviceID: %d)",
				threadUtilization, gpuLimit.ActiveThreadPercentage, gpuLimit.DeviceID)
			threadUtilization = gpuLimit.ActiveThreadPercentage
		}

		// Ensure thread utilization is within valid range
		if threadUtilization < 0 {
			threadUtilization = 0
		}
		if threadUtilization > 100 {
			threadUtilization = 100
		}

		consumptions[i] = ResourceConsumption{
			ActiveThreadPercentage: threadUtilization,
			MemoryUsageBytes:       memoryUsage,
			DeviceID:               gpuLimit.DeviceID,
		}
	}

	return consumptions
}

// estimateThreadOccupancy estimates GPU thread utilization
func (r *queueTheoryResourceCalculator) estimateThreadOccupancy(effectiveBatch, prefillTokens float64, numGPU int) float64 {
	// Determine Total Parallelism Demand
	// For LLMs, threads are typically assigned per hidden dimension.
	// Prefill phase: Uses hiddenSize threads (processes tokens in parallel via matrix ops)
	// Decode phase: Uses hiddenSize threads per sequence in the batch
	prefillThreads := float64(r.complexity.MinThreads)
	decodeThreads := effectiveBatch * float64(r.complexity.MinThreads)

	// In a single vLLM iteration, the GPU handles both.
	// Use max instead of sum since they may not run simultaneously
	totalActiveThreads := math.Max(prefillThreads, decodeThreads)

	// Compare against hardware capacity
	occupancy := (totalActiveThreads / float64(r.gpuSpec.MaxThreads*numGPU)) * 100

	// Cap at 100%
	if occupancy > 100.0 {
		occupancy = 100.0
	}

	return occupancy
}

// estimateMemoryUsage estimates memory usage in GB
func (r *queueTheoryResourceCalculator) estimateMemoryUsage(promptLen int, outputLen int, prefillTokens float64, numGPU int) float64 {
	totalTokens := float64(promptLen+outputLen) * float64(r.effectiveBatchSize)
	modelWeight := r.modelSize / float64(numGPU)
	kvCacheMem := (float64(r.effectiveBatchSize) * totalTokens * r.memoryPerToken) / float64(numGPU) / 1e9
	activationMem := r.activeMemoryPerPrefillToken * float64(prefillTokens) / float64(numGPU) / 1e9

	// Add attention memory for long contexts
	attentionMem := r.estimateAttentionMemory(promptLen, prefillTokens) / float64(numGPU)

	totalMemoryGBPerGPU := (modelWeight + kvCacheMem + activationMem + attentionMem) * r.memoryOverheadFactor

	return totalMemoryGBPerGPU
}

// estimateAttentionMemory calculates additional memory needed for attention computation
func (r *queueTheoryResourceCalculator) estimateAttentionMemory(seqLen int, prefillTokens float64) float64 {
	// For sequences <= 8K tokens, FlashAttention makes attention memory negligible
	if seqLen <= 8192 {
		return 0
	}

	// For long contexts, add a sublinear scaling factor
	longContextFactor := math.Pow(float64(seqLen)/8192.0, 1.3)

	// Estimate attention overhead
	attentionOverhead := prefillTokens * float64(r.params.HiddenSize) * float64(r.params.Precision) * longContextFactor * 0.1

	return attentionOverhead / 1e9 // Convert to GB
}

// GetGPULimits returns the configured GPU resource limits
func (r *queueTheoryResourceCalculator) GetGPULimits() []GPUResourceLimits {
	return r.gpuLimits
}

// estimateModelComplexity estimates model complexity metrics
func estimateModelComplexity(_ string, paramsInBillions float64, hiddenSize int) ComplexityMetrics {
	// Calculate FLOPs per token
	flops := 2 * paramsInBillions * 1e9

	// Arithmetic Intensity
	ai := 1.0

	// Thread Utilization
	minThreads := hiddenSize

	return ComplexityMetrics{
		FlopsPerToken:       flops,
		ArithmeticIntensity: ai,
		MinThreads:          minThreads,
	}
}

// inferredArchParams infers architecture parameters from model ID
func inferredArchParams(modelID string, logger logr.Logger) (bool, ArchParams) {
	id := strings.ToLower(modelID)
	// Check for specific known families
	for key, p := range modelFamilyMap {
		if strings.Contains(id, key) {
			return true, p
		}
	}
	_, paramsInBillions := extractParams(modelID, logger)
	precisionBytes := inferredPrecisionBytes(modelID)

	return false, ArchParams{ParametersB: paramsInBillions, Layers: 32, KVHeads: 8, HiddenSize: 4096, QueryHeads: 32, Precision: precisionBytes}
}

// estimateModelSizeQT calculates the VRAM needed to load the model weights
func estimateModelSizeQT(_ string, paramsInBillions float64, precisionBytes int) float64 {
	return paramsInBillions * float64(precisionBytes)
}

// estimateMemoryPerTokenFromID calculates memory per token for KV cache
func estimateMemoryPerTokenFromID(params ArchParams, precisionBytes int) float64 {
	// Calculate Head Dim
	headDim := params.HiddenSize / params.QueryHeads

	// Formula: 2 (K+V) * Layers * KV_Heads * Head_Dim * Precision
	bytesPerToken := 2 * params.Layers * params.KVHeads * headDim * precisionBytes

	return float64(bytesPerToken)
}

// estimateActiveMemoryPerTokenFromID calculates active memory per token
func estimateActiveMemoryPerTokenFromID(effectiveBatch float64, params ArchParams, precisionBytes int) float64 {
	// Base activation memory
	baseActivation := effectiveBatch * float64(params.HiddenSize*precisionBytes)

	// FFN intermediate activations
	ffnIntermediate := effectiveBatch * float64(params.HiddenSize*4*precisionBytes)

	return baseActivation + ffnIntermediate
}

// inferredPrecisionBytes infers precision in bytes from model ID
func inferredPrecisionBytes(modelID string) int {
	id := strings.ToLower(modelID)

	if strings.Contains(id, "fp8") || strings.Contains(id, "int8") {
		return 1
	}

	if strings.Contains(id, "fp16") || strings.Contains(id, "bf16") || strings.Contains(id, "half") {
		return 2
	}

	if strings.Contains(id, "awq") || strings.Contains(id, "gptq") || strings.Contains(id, "4bit") {
		return 2
	}

	// Default for most modern LLMs
	return 2
}

// extractParams extracts parameter count from model ID
func extractParams(modelID string, logger logr.Logger) (bool, float64) {
	id := strings.ToLower(modelID)
	re := regexp.MustCompile(`(\d+\.?\d*)b`)
	matches := re.FindStringSubmatch(id)

	if len(matches) > 1 {
		val, err := strconv.ParseFloat(matches[1], 64)
		if err == nil {
			return true, val
		}
	}
	return false, 8.0 // Default to 8B if unknown
}

// getGPUSpecs retrieves GPU specifications
func getGPUSpecs(name string, logger logr.Logger) (bool, *GPUSpecs) {
	name = strings.ToLower(name)
	for key, spec := range gpuRegistry {
		if strings.Contains(name, key) {
			return true, &spec
		}
	}
	spec := gpuRegistry["a100"]
	return false, &spec
}

// parseGPULimitsFromEnv parses GPU resource limits from environment variables
func parseGPULimitsFromEnv() ([]GPUResourceLimits, error) {
	limits := make(map[int]*GPUResourceLimits)

	// Scan environment variables
	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := parts[0]
		value := parts[1]

		// Check for GPU_DEVICE_<N>_ACTIVE_THREAD_PERCENTAGE
		if strings.HasPrefix(key, "GPU_DEVICE_") && strings.HasSuffix(key, "_ACTIVE_THREAD_PERCENTAGE") {
			deviceIDStr := strings.TrimPrefix(key, "GPU_DEVICE_")
			deviceIDStr = strings.TrimSuffix(deviceIDStr, "_ACTIVE_THREAD_PERCENTAGE")
			deviceID, err := strconv.Atoi(deviceIDStr)
			if err != nil {
				return nil, fmt.Errorf("invalid device ID in %s: %w", key, err)
			}

			percentage, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid percentage value in %s: %w", key, err)
			}

			if percentage < 0 || percentage > 100 {
				return nil, fmt.Errorf("percentage must be between 0 and 100 in %s", key)
			}

			if limits[deviceID] == nil {
				limits[deviceID] = &GPUResourceLimits{DeviceID: deviceID}
			}
			limits[deviceID].ActiveThreadPercentage = percentage
		}

		// Check for GPU_DEVICE_<N>_MEMORY_LIMIT
		if strings.HasPrefix(key, "GPU_DEVICE_") && strings.HasSuffix(key, "_MEMORY_LIMIT") {
			deviceIDStr := strings.TrimPrefix(key, "GPU_DEVICE_")
			deviceIDStr = strings.TrimSuffix(deviceIDStr, "_MEMORY_LIMIT")
			deviceID, err := strconv.Atoi(deviceIDStr)
			if err != nil {
				return nil, fmt.Errorf("invalid device ID in %s: %w", key, err)
			}

			memoryBytes, err := parseMemorySize(value)
			if err != nil {
				return nil, fmt.Errorf("invalid memory size in %s: %w", key, err)
			}

			if limits[deviceID] == nil {
				limits[deviceID] = &GPUResourceLimits{DeviceID: deviceID}
			}
			limits[deviceID].MemoryLimitBytes = memoryBytes
		}
	}

	// Convert map to slice and validate
	result := make([]GPUResourceLimits, 0, len(limits))
	for _, limit := range limits {
		// Validate that both fields are set
		if limit.ActiveThreadPercentage == 0 && limit.MemoryLimitBytes == 0 {
			return nil, fmt.Errorf("GPU device %d has no limits configured", limit.DeviceID)
		}
		result = append(result, *limit)
	}

	return result, nil
}

// parseMemorySize parses memory size strings like "12Gi", "8GB", "1024Mi", etc.
func parseMemorySize(size string) (int64, error) {
	size = strings.TrimSpace(size)
	if size == "" {
		return 0, fmt.Errorf("empty memory size")
	}

	// Find where the number ends and unit begins
	var numStr string
	var unit string
	foundUnit := false
	for i, ch := range size {
		if (ch < '0' || ch > '9') && ch != '.' {
			numStr = size[:i]
			unit = size[i:]
			foundUnit = true
			break
		}
	}

	// If no unit found, the entire string is the number
	if !foundUnit {
		numStr = size
		unit = ""
	}

	if numStr == "" {
		return 0, fmt.Errorf("no numeric value found in memory size: %s", size)
	}

	value, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid numeric value: %w", err)
	}

	// Parse unit (case-insensitive)
	unit = strings.ToUpper(strings.TrimSpace(unit))
	var multiplier int64

	switch unit {
	case "B", "":
		multiplier = 1
	case "K", "KB":
		multiplier = 1000
	case "KI", "KIB":
		multiplier = 1024
	case "M", "MB":
		multiplier = 1000 * 1000
	case "MI", "MIB":
		multiplier = 1024 * 1024
	case "G", "GB":
		multiplier = 1000 * 1000 * 1000
	case "GI", "GIB":
		multiplier = 1024 * 1024 * 1024
	case "T", "TB":
		multiplier = 1000 * 1000 * 1000 * 1000
	case "TI", "TIB":
		multiplier = 1024 * 1024 * 1024 * 1024
	default:
		return 0, fmt.Errorf("unknown memory unit: %s", unit)
	}

	return int64(value * float64(multiplier)), nil
}
