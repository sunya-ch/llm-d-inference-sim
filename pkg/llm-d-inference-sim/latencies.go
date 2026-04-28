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
	"os"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

type TTFTParams struct {
	PromptTokens         int
	CachedPromptTokens   int
	DoRemotePrefill      bool
	RunningReqs          int64
	ThreadUtilization    float64 // Current GPU thread utilization percentage (0-100)
	ThreadUtilizationCap float64 // GPU thread utilization limit/cap (0-100)
}

type InterTokenParams struct {
	RunningReqs          int64
	ThreadUtilization    float64 // Current GPU thread utilization percentage (0-100)
	ThreadUtilizationCap float64 // GPU thread utilization limit/cap (0-100)
}

type LatencyCalculator interface {
	// GetTimeToFirstToken returns time to first token. The simulator will wait
	// this amount of time before generating the first token.
	GetTimeToFirstToken(params *TTFTParams) time.Duration
	// GetInterTokenLatency returns inter-token latency. The simulator will wait
	// this amount of time before generating each response token (except the first one).
	GetInterTokenLatency(params *InterTokenParams) time.Duration
}

type baseCalculator struct {
	interTokenLatency       time.Duration
	interTokenLatencyStdDev time.Duration
	timeFactorUnderLoad     float64
	maxNumSeqs              int
	random                  *common.Random
	gpuSpec                 *GPUSpecs
}

// returns inter token latency
func (b *baseCalculator) GetInterTokenLatency(params *InterTokenParams) time.Duration {
	allocationImpact := b.CalculateResourceAllocationImpact(params.ThreadUtilization, params.ThreadUtilizationCap)
	latency := time.Duration(float64(b.interTokenLatency) * b.getCurrLoadFactor(params.RunningReqs) * (1.0 + allocationImpact))
	return b.random.RandomNormDuration(latency, b.interTokenLatencyStdDev)
}

func (b *baseCalculator) getCurrLoadFactor(nRunningReqs int64) float64 {
	if b.maxNumSeqs <= 1 {
		return 1.0
	}
	return 1 + (b.timeFactorUnderLoad-1)*float64(nRunningReqs-1)/float64(b.maxNumSeqs-1)
}

func (b *baseCalculator) CalculateResourceAllocationImpact(utilization, cap float64) float64 {
	// If no GPU specs provided, return 0.0 (no adjustment)
	if b.gpuSpec == nil {
		return 0.0
	}
	// If no utilization data provided, return 1.0 (no adjustment)
	if cap <= 0 || utilization <= 0 {
		return 0.0
	}

	ratio := utilization / cap

	if ratio < 0.0 {
		ratio = 0.0
	}
	if ratio > 1.0 {
		ratio = 1.0
	}
	// 1. Under-provisioned (Low Density): Apply Speedup
	// We call this 'Overprovisioning the work' to get a speedup
	if ratio < 0.5 {
		// As the ratio gets smaller, the potential 'Speedup' bonus is higher
		// (1.0 - ratio) represents the 'empty space' we can fill
		return -(1.0 - ratio) * b.gpuSpec.OverprovisionSpeedUp
	}

	// 2. Over-provisioned (High Density): Apply Contention
	// We call this 'Under-provisioning the hardware' relative to the load
	if ratio > 0.8 {
		// As the ratio approaches 1.0 (or exceeds it via queuing),
		// the contention penalty scales up.
		return (ratio - 0.8) * b.gpuSpec.UnderprovisionContentionImpact
	}
	return 0.0
}

// Default latency calculator. Decides whether to use per token or
// constant latency calculations based on the values of time-to-first-token
// and kv-cache-transfer-latency.
type defaultCalculator struct {
	baseCalculator
	timeToFirstToken             time.Duration
	timeToFirstTokenStdDev       time.Duration
	kVCacheTransferLatency       time.Duration
	kVCacheTransferLatencyStdDev time.Duration
	kVCacheTransferTimePerToken  time.Duration
	kVCacheTransferTimeStdDev    time.Duration
	prefillOverhead              time.Duration
	prefillTimePerToken          time.Duration
	prefillTimeStdDev            time.Duration
}

func newDefaultCalculator(config *common.Configuration, random *common.Random) *defaultCalculator {
	// Get GPU name from environment or use default
	gpuName := os.Getenv("GPU_MODEL_NAME_0")
	if gpuName == "" {
		gpuName = "NVIDIA-A100-SXM4-80GB" // default
	}
	// Get GPU specs
	gpuFound, gpuSpec := getGPUSpecs(gpuName, logr.Logger{})
	if !gpuFound {
		_, gpuSpec = getGPUSpecs("NVIDIA-A100-SXM4-80GB", logr.Logger{})
	}
	return &defaultCalculator{
		baseCalculator: baseCalculator{
			interTokenLatency:       config.InterTokenLatency.ToDuration(),
			interTokenLatencyStdDev: config.InterTokenLatencyStdDev.ToDuration(),
			timeFactorUnderLoad:     config.TimeFactorUnderLoad,
			maxNumSeqs:              config.MaxNumSeqs,
			random:                  random,
			gpuSpec:                 gpuSpec,
		},
		timeToFirstToken:             config.TimeToFirstToken.ToDuration(),
		timeToFirstTokenStdDev:       config.TimeToFirstTokenStdDev.ToDuration(),
		kVCacheTransferLatency:       config.KVCacheTransferLatency.ToDuration(),
		kVCacheTransferLatencyStdDev: config.KVCacheTransferLatencyStdDev.ToDuration(),
		kVCacheTransferTimePerToken:  config.KVCacheTransferTimePerToken.ToDuration(),
		kVCacheTransferTimeStdDev:    config.KVCacheTransferTimeStdDev.ToDuration(),
		prefillOverhead:              config.PrefillOverhead.ToDuration(),
		prefillTimePerToken:          config.PrefillTimePerToken.ToDuration(),
		prefillTimeStdDev:            config.PrefillTimeStdDev.ToDuration(),
	}
}

// returns time to first token
func (d *defaultCalculator) GetTimeToFirstToken(params *TTFTParams) time.Duration {
	if params.DoRemotePrefill {
		if d.kVCacheTransferLatency == 0 && d.kVCacheTransferLatencyStdDev == 0 {
			// is disaggregated PD and ttft is calculated using number of prompt tokens
			kvCacheTransT := d.kVCacheTransferTimePerToken * time.Duration(params.PromptTokens)
			return d.random.RandomNormDuration(kvCacheTransT, d.kVCacheTransferTimeStdDev)
		}
		// is disaggregated PD and *not* using number of prompt tokens
		return d.random.RandomNormDuration(d.kVCacheTransferLatency, d.kVCacheTransferLatencyStdDev)
	}
	if d.timeToFirstToken == 0 && d.timeToFirstTokenStdDev == 0 {
		// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
		prefillOverhead := time.Duration(float64(d.prefillOverhead) * d.getCurrLoadFactor(params.RunningReqs))
		allocationImpact := d.CalculateResourceAllocationImpact(params.ThreadUtilization, params.ThreadUtilizationCap)
		prefillTimePerToken := time.Duration(float64(d.prefillTimePerToken) * d.getCurrLoadFactor(params.RunningReqs) * (1.0 + allocationImpact))
		prefillTime := prefillOverhead + time.Duration(params.PromptTokens-params.CachedPromptTokens)*prefillTimePerToken
		return d.random.RandomNormDuration(prefillTime, d.prefillTimeStdDev)
	}

	// is aggregated PD and *not* using number of prompt tokens
	ttft := time.Duration(float64(d.timeToFirstToken) * d.getCurrLoadFactor(params.RunningReqs))
	return d.random.RandomNormDuration(ttft, d.timeToFirstTokenStdDev)
}

// Constant latency calculator doesn't take the prompt size into account, uses
// time-to-first-token and kv-cache-transfer-latency and their std devs.
type constantCalculator struct {
	baseCalculator
	timeToFirstToken             time.Duration
	timeToFirstTokenStdDev       time.Duration
	kVCacheTransferLatency       time.Duration
	kVCacheTransferLatencyStdDev time.Duration
}

func newConstantCalculator(config *common.Configuration, random *common.Random) *constantCalculator {
	return &constantCalculator{
		baseCalculator: baseCalculator{
			interTokenLatency:       config.InterTokenLatency.ToDuration(),
			interTokenLatencyStdDev: config.InterTokenLatencyStdDev.ToDuration(),
			timeFactorUnderLoad:     config.TimeFactorUnderLoad,
			maxNumSeqs:              config.MaxNumSeqs,
			random:                  random,
		},
		timeToFirstToken:             config.TimeToFirstToken.ToDuration(),
		timeToFirstTokenStdDev:       config.TimeToFirstTokenStdDev.ToDuration(),
		kVCacheTransferLatency:       config.KVCacheTransferLatency.ToDuration(),
		kVCacheTransferLatencyStdDev: config.KVCacheTransferLatencyStdDev.ToDuration(),
	}
}

// returns time to first token
func (c *constantCalculator) GetTimeToFirstToken(params *TTFTParams) time.Duration {
	if params.DoRemotePrefill {
		// is disaggregated PD and *not* using number of prompt tokens
		return c.random.RandomNormDuration(c.kVCacheTransferLatency, c.kVCacheTransferLatencyStdDev)
	}
	// is aggregated PD and *not* using number of prompt tokens
	ttft := time.Duration(float64(c.timeToFirstToken) * c.getCurrLoadFactor(params.RunningReqs))
	return c.random.RandomNormDuration(ttft, c.timeToFirstTokenStdDev)
}

// Per token calculator takes the prompt size into account
type perTokenCalculator struct {
	baseCalculator
	kVCacheTransferTimePerToken time.Duration
	kVCacheTransferTimeStdDev   time.Duration
	prefillOverhead             time.Duration
	prefillTimePerToken         time.Duration
	prefillTimeStdDev           time.Duration
}

func newPerTokenCalculator(config *common.Configuration, random *common.Random) *perTokenCalculator {
	return &perTokenCalculator{
		baseCalculator: baseCalculator{
			interTokenLatency:       config.InterTokenLatency.ToDuration(),
			interTokenLatencyStdDev: config.InterTokenLatencyStdDev.ToDuration(),
			timeFactorUnderLoad:     config.TimeFactorUnderLoad,
			maxNumSeqs:              config.MaxNumSeqs,
			random:                  random,
		},
		kVCacheTransferTimePerToken: config.KVCacheTransferTimePerToken.ToDuration(),
		kVCacheTransferTimeStdDev:   config.KVCacheTransferTimeStdDev.ToDuration(),
		prefillOverhead:             config.PrefillOverhead.ToDuration(),
		prefillTimePerToken:         config.PrefillTimePerToken.ToDuration(),
		prefillTimeStdDev:           config.PrefillTimeStdDev.ToDuration(),
	}
}

// returns time to first token
func (p *perTokenCalculator) GetTimeToFirstToken(params *TTFTParams) time.Duration {
	if params.DoRemotePrefill {
		// is disaggregated PD and ttft is calculated using number of prompt tokens
		kvCacheTransT := p.kVCacheTransferTimePerToken * time.Duration(params.PromptTokens)
		return p.random.RandomNormDuration(kvCacheTransT, p.kVCacheTransferTimeStdDev)
	}
	// is aggregated PD and ttft is calculated using number of prompt tokens that are not in kv cache
	prefillOverhead := time.Duration(float64(p.prefillOverhead) * p.getCurrLoadFactor(params.RunningReqs))
	prefillTimePerToken := time.Duration(float64(p.prefillTimePerToken) * p.getCurrLoadFactor(params.RunningReqs))
	prefillTime := prefillOverhead + time.Duration(params.PromptTokens-params.CachedPromptTokens)*prefillTimePerToken
	return p.random.RandomNormDuration(prefillTime, p.prefillTimeStdDev)
}
