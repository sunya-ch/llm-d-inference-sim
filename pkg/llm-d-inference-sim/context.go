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
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	kvcache "github.com/llm-d/llm-d-inference-sim/pkg/kv-cache"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	vllmapi "github.com/llm-d/llm-d-inference-sim/pkg/vllm-api"
)

// LoRAs usage info for requests execution
type lorasUsageInfo struct {
	mux sync.RWMutex
	// lora adapter name -> reference count (number of currently running requests)
	loadedLoras map[string]int
	// channel for "there is a LoRA that can be removed" event
	loraRemovable common.Channel[int]
	// maximum number of LoRAs that can be used simultaneously
	maxLoras int
}

type SimContext struct {
	// logger is used for information and errors logging
	logger logr.Logger
	// metrics contains all Prometheus metrics related data
	metrics metricsData
	// Config is the simulator's configuration
	Config *common.Configuration
	// loraAdaptors contains list of LoRA available adaptors
	loraAdaptors sync.Map
	// loras contains information about which LoRAs are in use
	loras *lorasUsageInfo
	// rand with a configurable seed to generate reproducible Random responses
	Random *common.Random
	// kv cache functionality
	kvcacheHelper *kvcache.KVCacheHelper
	// dataset is used for token generation in responses
	dataset dataset.Dataset
	// latencyCalculator calculates the delays in simulator's responses
	latencyCalculator LatencyCalculator
	// Tokenizer used for request tokenization and in /tokenize
	Tokenizer tokenizer.Tokenizer
	// resourceCalculator estimates GPU resource consumption
	resourceCalculator ResourceCalculator
	// currentKVCacheUsage tracks the current KV cache utilization (0-1)
	currentKVCacheUsage float64
	// kvCacheUsageMux protects currentKVCacheUsage
	kvCacheUsageMux sync.RWMutex
}

func (s *SimContext) initialize(ctx context.Context) error {
	s.Random = common.NewRandom(s.Config.Seed, s.Config.Port)

	switch s.Config.LatencyCalculator {
	case common.DefaultLatencyCalculator:
		s.latencyCalculator = newDefaultCalculator(s.Config, s.Random)
	case common.ConstantLatencyCalculator:
		s.latencyCalculator = newConstantCalculator(s.Config, s.Random)
	case common.PerPromptTokenLatencyCalculator:
		s.latencyCalculator = newPerTokenCalculator(s.Config, s.Random)
	}

	// Initialize resource calculator
	resourceCalc, err := NewResourceCalculator(s.Config)
	if err != nil {
		return fmt.Errorf("failed to create resource calculator: %w", err)
	}
	s.resourceCalculator = resourceCalc
	if s.resourceCalculator != nil {
		s.logger.V(logging.INFO).Info("GPU resource tracking enabled",
			"devices", len(s.resourceCalculator.GetGPULimits()))
	}

	for _, lora := range s.Config.LoraModules {
		s.loraAdaptors.Store(lora.Name, "")
	}
	s.loras.maxLoras = s.Config.MaxLoras
	s.loras.loraRemovable = common.Channel[int]{
		Channel: make(chan int, s.Config.MaxNumSeqs),
		Name:    "loraRemovable",
	}

	// initialize prometheus metrics
	err = s.createAndRegisterPrometheus(ctx)
	if err != nil {
		return err
	}

	// KVCache doesn't support images at the moment, so in mm-encoder only mode
	// we don't start it.
	if s.Config.EnableKVCache && !s.Config.MMEncoderOnly {
		s.kvcacheHelper, err = kvcache.NewKVCacheHelper(ctx, s.Config, s.logger,
			s.metrics.kvCacheUsageChan, s.metrics.prefixCacheStatsChan, s.Tokenizer)
		if err != nil {
			return err
		}

		go s.kvcacheHelper.Run(ctx)
	}

	err = s.initDataset(ctx)
	if err != nil {
		return fmt.Errorf("dataset initialization error: %w", err)
	}

	return nil
}

func (s *SimContext) initDataset(ctx context.Context) error {
	if s.Config.MMEncoderOnly {
		var err error
		s.dataset, err = dataset.NewMMEncoderOnlyDataset(s.logger, s.Tokenizer)
		if err != nil {
			return fmt.Errorf("failed to initialize dataset for mm-encoder-only mode: %w", err)
		}
		return nil
	}

	if s.Config.Mode == common.ModeEcho {
		s.dataset = &dataset.EchoDataset{}
		return nil
	}

	if s.Config.DatasetPath == "" && s.Config.DatasetURL == "" {
		// use predefined sentences as responses
		randDataset := &dataset.DefaultDataset{}
		err := randDataset.Init(ctx, s.logger, s.Random, s.Config.MaxModelLen, s.Tokenizer)
		if err != nil {
			return fmt.Errorf("failed to initialize random dataset: %w", err)
		}
		s.logger.V(logging.INFO).Info("No dataset path or URL provided, using random text for responses")
		s.dataset = randDataset
		return nil
	}

	// use dataset containing responses
	custDataset := &dataset.CustomDataset{}
	err := custDataset.Init(ctx, s.logger, s.Random, s.Config.DatasetPath, s.Config.DatasetTableName,
		s.Config.DatasetInMemory, s.Config.MaxModelLen, s.Tokenizer)

	if err == nil {
		s.dataset = custDataset
		return nil
	}

	return err
}

// isLora returns true if the given model name is one of loaded LoRAs
func (s *SimContext) isLora(model string) bool {
	for _, lora := range s.getLoras() {
		if model == lora {
			return true
		}
	}

	return false
}

// getDisplayedModelName returns the model name that must appear in API
// responses.  LoRA adapters keep their explicit name, while all base-model
// requests are surfaced as the first alias from --served-model-name.
func (s *SimContext) getDisplayedModelName(reqModel string) string {
	if s.isLora(reqModel) {
		return reqModel
	}
	return s.Config.ServedModelNames[0]
}

func (s *SimContext) simulateTTFT(reqCtx requestContext, respCtx ResponseContext) {
	startPrefill := time.Now()

	// Get current thread utilization and cap from resource calculator
	threadUtilization := 0.0
	threadUtilizationCap := 0.0
	if s.resourceCalculator != nil {
		// Calculate resource consumption to get current utilization
		consumptions := s.calculateResourceConsumption(
			respCtx.UsageData().PromptTokens,
			respCtx.UsageData().CompletionTokens,
			respCtx.NumberCachedPromptTokens(),
			s.Config.Model,
		)

		// Use average thread utilization across all GPUs
		if len(consumptions) > 0 {
			totalThreadUtil := 0.0
			for _, c := range consumptions {
				totalThreadUtil += c.ActiveThreadPercentage
			}
			threadUtilization = totalThreadUtil / float64(len(consumptions))
		}

		// Get the GPU limit/cap (use average across all GPUs)
		gpuLimits := s.resourceCalculator.GetGPULimits()
		if len(gpuLimits) > 0 {
			totalCap := 0.0
			for _, limit := range gpuLimits {
				totalCap += limit.ActiveThreadPercentage
			}
			threadUtilizationCap = totalCap / float64(len(gpuLimits))
		}
	}

	// time to first token delay
	params := TTFTParams{
		PromptTokens:         respCtx.UsageData().PromptTokens,
		CachedPromptTokens:   respCtx.NumberCachedPromptTokens(),
		DoRemotePrefill:      respCtx.doRemotePrefill(),
		RunningReqs:          s.metrics.nRunningReqs,
		ThreadUtilization:    threadUtilization,
		ThreadUtilizationCap: threadUtilizationCap,
	}
	ttft := s.latencyCalculator.GetTimeToFirstToken(&params)
	time.Sleep(ttft)
	// Report TTFT including queue time (time from request arrival to first token)
	// This matches industry standard where TTFT includes all time user experiences
	ttftWithQueue := time.Since(reqCtx.startProcessingTime())
	common.WriteToChannel(s.metrics.ttftChan, ttftWithQueue.Seconds(), s.logger)
	common.WriteToChannel(s.metrics.reqPrefillTimeChan, time.Since(startPrefill).Seconds(), s.logger)
}

func (s *SimContext) simulateInterTokenLatency() {
	// Get current thread utilization and cap from resource calculator
	threadUtilization := 0.0
	threadUtilizationCap := 0.0
	if s.resourceCalculator != nil {
		// For inter-token latency, we need to estimate based on current state
		// Use a simplified calculation with minimal tokens
		consumptions := s.calculateResourceConsumption(1, 1, 0, s.Config.Model)

		// Use average thread utilization across all GPUs
		if len(consumptions) > 0 {
			totalThreadUtil := 0.0
			for _, c := range consumptions {
				totalThreadUtil += c.ActiveThreadPercentage
			}
			threadUtilization = totalThreadUtil / float64(len(consumptions))
		}

		// Get the GPU limit/cap (use average across all GPUs)
		gpuLimits := s.resourceCalculator.GetGPULimits()
		if len(gpuLimits) > 0 {
			totalCap := 0.0
			for _, limit := range gpuLimits {
				totalCap += limit.ActiveThreadPercentage
			}
			threadUtilizationCap = totalCap / float64(len(gpuLimits))
		}
	}

	perTokenLatency := s.latencyCalculator.GetInterTokenLatency(&InterTokenParams{
		RunningReqs:          s.metrics.nRunningReqs,
		ThreadUtilization:    threadUtilization,
		ThreadUtilizationCap: threadUtilizationCap,
	})
	time.Sleep(perTokenLatency)

	// report tpot in seconds
	common.WriteToChannel(s.metrics.tpotChan, perTokenLatency.Seconds(), s.logger)
}

// calculateResourceConsumption estimates GPU resource consumption for a request
func (s *SimContext) calculateResourceConsumption(promptTokens, generationTokens, cachedPromptTokens int, modelName string) []ResourceConsumption {
	if s.resourceCalculator == nil {
		return []ResourceConsumption{}
	}

	// Get current KV cache usage
	s.kvCacheUsageMux.RLock()
	kvCacheUsage := s.currentKVCacheUsage
	s.kvCacheUsageMux.RUnlock()

	// Calculate KV cache size in tokens
	// KVCacheSize is in blocks, need to convert to tokens
	kvCacheSizeTokens := 0
	if s.Config.EnableKVCache && s.Config.KVCacheSize > 0 {
		// Each block contains TokenBlockSize tokens
		kvCacheSizeTokens = s.Config.KVCacheSize * s.Config.TokenBlockSize
	}

	params := &ResourceParams{
		PromptTokens:           promptTokens,
		GenerationTokens:       generationTokens,
		CachedPromptTokens:     cachedPromptTokens,
		RunningReqs:            s.metrics.nRunningReqs,
		ModelName:              modelName,
		KVCacheUsagePercentage: kvCacheUsage,
		KVCacheSizeTokens:      kvCacheSizeTokens,
	}

	consumptions := s.resourceCalculator.CalculateResourceConsumption(params)

	// Log resource consumption for each GPU
	for _, consumption := range consumptions {
		s.logger.V(logging.DEBUG).Info("Estimated resource consumption",
			"model", modelName,
			"promptTokens", promptTokens,
			"generationTokens", generationTokens,
			"cachedTokens", cachedPromptTokens,
			"kvCacheUsage", fmt.Sprintf("%.2f%%", kvCacheUsage*100),
			"kvCacheSizeTokens", kvCacheSizeTokens,
			"deviceID", consumption.DeviceID,
			"threadPercentage", fmt.Sprintf("%.2f%%", consumption.ActiveThreadPercentage),
			"memoryMB", consumption.MemoryUsageBytes/(1024*1024))
	}

	return consumptions
}

// updateKVCacheUsage updates the current KV cache usage percentage
func (s *SimContext) updateKVCacheUsage(usage float64) {
	s.kvCacheUsageMux.Lock()
	s.currentKVCacheUsage = usage
	s.kvCacheUsageMux.Unlock()
}

// CreateModelsResponse creates and returns ModelResponse for the current state, returned array of models contains the base model + LoRA adapters if exist
func (s *SimContext) CreateModelsResponse() *vllmapi.ModelsResponse {
	modelsResp := vllmapi.ModelsResponse{Object: "list", Data: []vllmapi.ModelsResponseModelInfo{}}

	// Advertise every public model alias
	for _, alias := range s.Config.ServedModelNames {
		modelsResp.Data = append(modelsResp.Data, vllmapi.ModelsResponseModelInfo{
			ID:      alias,
			Object:  vllmapi.ObjectModel,
			Created: time.Now().Unix(),
			OwnedBy: "vllm",
			Root:    alias,
			Parent:  nil,
		})
	}

	// add LoRA adapter's info
	parent := s.Config.ServedModelNames[0]
	for _, lora := range s.getLoras() {
		modelsResp.Data = append(modelsResp.Data, vllmapi.ModelsResponseModelInfo{
			ID:      lora,
			Object:  vllmapi.ObjectModel,
			Created: time.Now().Unix(),
			OwnedBy: "vllm",
			Root:    lora,
			Parent:  &parent,
		})
	}

	return &modelsResp
}
