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

package tests

import (
	"os"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	llmdinferencesim "github.com/llm-d/llm-d-inference-sim/pkg/llm-d-inference-sim"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("GPU Resource Calculator", func() {
	var originalEnv []string

	BeforeEach(func() {
		// Save original environment
		originalEnv = os.Environ()
	})

	AfterEach(func() {
		// Restore original environment
		os.Clearenv()
		for _, env := range originalEnv {
			if key, val, ok := splitEnv(env); ok {
				os.Setenv(key, val)
			}
		}
	})

	Context("When GPU limits are configured", func() {
		It("should create a resource calculator with single GPU", func() {
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "50")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "12Gi")

			config := &common.Configuration{}
			calc, err := llmdinferencesim.NewResourceCalculator(config)

			Expect(err).NotTo(HaveOccurred())
			Expect(calc).NotTo(BeNil())

			limits := calc.GetGPULimits()
			Expect(limits).To(HaveLen(1))
			Expect(limits[0].DeviceID).To(Equal(0))
			Expect(limits[0].ActiveThreadPercentage).To(Equal(50.0))
			Expect(limits[0].MemoryLimitBytes).To(Equal(int64(12884901888))) // 12Gi
		})

		It("should create a resource calculator with multiple GPUs", func() {
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "10")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "12Gi")
			os.Setenv("GPU_DEVICE_1_ACTIVE_THREAD_PERCENTAGE", "15")
			os.Setenv("GPU_DEVICE_1_MEMORY_LIMIT", "16Gi")

			config := &common.Configuration{}
			calc, err := llmdinferencesim.NewResourceCalculator(config)

			Expect(err).NotTo(HaveOccurred())
			Expect(calc).NotTo(BeNil())

			limits := calc.GetGPULimits()
			Expect(limits).To(HaveLen(2))
		})

		It("should calculate resource consumption correctly", func() {
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "50")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "12Gi")

			config := &common.Configuration{}
			calc, err := llmdinferencesim.NewResourceCalculator(config)
			Expect(err).NotTo(HaveOccurred())

			params := &llmdinferencesim.ResourceParams{
				PromptTokens:       100,
				GenerationTokens:   50,
				CachedPromptTokens: 20,
				RunningReqs:        1,
				ModelName:          "test-model",
			}

			consumptions := calc.CalculateResourceConsumption(params)

			Expect(consumptions).To(HaveLen(1))
			Expect(consumptions[0].DeviceID).To(Equal(0))
			Expect(consumptions[0].ActiveThreadPercentage).To(BeNumerically(">", 0))
			Expect(consumptions[0].ActiveThreadPercentage).To(BeNumerically("<=", 50))
			Expect(consumptions[0].MemoryUsageBytes).To(BeNumerically(">", 0))
		})

		It("should increase thread utilization with more concurrent requests", func() {
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "50")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "12Gi")

			config := &common.Configuration{}
			calc, err := llmdinferencesim.NewResourceCalculator(config)
			Expect(err).NotTo(HaveOccurred())

			params1 := &llmdinferencesim.ResourceParams{
				PromptTokens:       100,
				GenerationTokens:   50,
				CachedPromptTokens: 20,
				RunningReqs:        1,
				ModelName:          "test-model",
			}

			params5 := &llmdinferencesim.ResourceParams{
				PromptTokens:       100,
				GenerationTokens:   50,
				CachedPromptTokens: 20,
				RunningReqs:        5,
				ModelName:          "test-model",
			}

			consumptions1 := calc.CalculateResourceConsumption(params1)
			consumptions5 := calc.CalculateResourceConsumption(params5)

			Expect(consumptions1).To(HaveLen(1))
			Expect(consumptions5).To(HaveLen(1))
			// With more concurrent requests, thread utilization should increase or stay the same
			// (it may be capped by GPU limits)
			Expect(consumptions5[0].ActiveThreadPercentage).To(BeNumerically(">=", consumptions1[0].ActiveThreadPercentage))
		})

		It("should distribute resources equally across multiple GPUs", func() {
			// Set up 2 GPUs with same limits
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "50")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "12Gi")
			os.Setenv("GPU_DEVICE_1_ACTIVE_THREAD_PERCENTAGE", "50")
			os.Setenv("GPU_DEVICE_1_MEMORY_LIMIT", "12Gi")

			config := &common.Configuration{Model: "test-7b"}
			calc, err := llmdinferencesim.NewResourceCalculator(config)
			Expect(err).NotTo(HaveOccurred())
			Expect(calc).NotTo(BeNil())

			// Verify we have 2 GPUs configured
			limits := calc.GetGPULimits()
			Expect(limits).To(HaveLen(2))

			params := &llmdinferencesim.ResourceParams{
				PromptTokens:       1000,
				GenerationTokens:   500,
				CachedPromptTokens: 0,
				RunningReqs:        1,
				ModelName:          "test-7b",
			}

			consumptions := calc.CalculateResourceConsumption(params)

			// Should return consumption for each GPU
			Expect(consumptions).To(HaveLen(2))

			// Verify resources are distributed equally
			gpu0 := consumptions[0]
			gpu1 := consumptions[1]

			// Memory should be equal across GPUs (within small tolerance for rounding)
			Expect(gpu0.MemoryUsageBytes).To(BeNumerically("~", gpu1.MemoryUsageBytes, 1024))

			// Thread utilization should be equal across GPUs
			Expect(gpu0.ActiveThreadPercentage).To(BeNumerically("~", gpu1.ActiveThreadPercentage, 0.1))

			// Each GPU should have valid device IDs
			Expect(gpu0.DeviceID).To(Or(Equal(0), Equal(1)))
			Expect(gpu1.DeviceID).To(Or(Equal(0), Equal(1)))
			Expect(gpu0.DeviceID).NotTo(Equal(gpu1.DeviceID))

			// Total resources should be reasonable
			totalMemory := gpu0.MemoryUsageBytes + gpu1.MemoryUsageBytes
			totalThreads := gpu0.ActiveThreadPercentage + gpu1.ActiveThreadPercentage

			Expect(totalMemory).To(BeNumerically(">", 0))
			Expect(totalThreads).To(BeNumerically(">", 0))
		})

		It("should respect per-GPU limits when distributing resources", func() {
			// Set up 2 GPUs with different limits
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "30")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "8Gi")
			os.Setenv("GPU_DEVICE_1_ACTIVE_THREAD_PERCENTAGE", "70")
			os.Setenv("GPU_DEVICE_1_MEMORY_LIMIT", "16Gi")

			config := &common.Configuration{Model: "test-7b"}
			calc, err := llmdinferencesim.NewResourceCalculator(config)
			Expect(err).NotTo(HaveOccurred())

			params := &llmdinferencesim.ResourceParams{
				PromptTokens:       2000,
				GenerationTokens:   1000,
				CachedPromptTokens: 0,
				RunningReqs:        5, // High load to trigger limits
				ModelName:          "test-7b",
			}

			consumptions := calc.CalculateResourceConsumption(params)
			Expect(consumptions).To(HaveLen(2))

			// Each GPU should respect its own limits
			for _, consumption := range consumptions {
				limits := calc.GetGPULimits()
				var gpuLimit llmdinferencesim.GPUResourceLimits
				for _, limit := range limits {
					if limit.DeviceID == consumption.DeviceID {
						gpuLimit = limit
						break
					}
				}

				// Thread utilization should not exceed the GPU's limit
				Expect(consumption.ActiveThreadPercentage).To(BeNumerically("<=", gpuLimit.ActiveThreadPercentage))
				// Memory should not exceed the GPU's limit
				Expect(consumption.MemoryUsageBytes).To(BeNumerically("<=", gpuLimit.MemoryLimitBytes))
			}
		})

		It("should handle invalid percentage gracefully", func() {
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "invalid")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "12Gi")

			config := &common.Configuration{}
			_, err := llmdinferencesim.NewResourceCalculator(config)

			Expect(err).To(HaveOccurred())
		})

		It("should handle invalid memory size gracefully", func() {
			os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "10")
			os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", "invalid")

			config := &common.Configuration{}
			_, err := llmdinferencesim.NewResourceCalculator(config)

			Expect(err).To(HaveOccurred())
		})
	})

	Context("When no GPU limits are configured", func() {
		It("should return nil calculator", func() {
			os.Clearenv()

			config := &common.Configuration{}
			calc, err := llmdinferencesim.NewResourceCalculator(config)

			Expect(err).NotTo(HaveOccurred())
			Expect(calc).To(BeNil())
		})
	})

	Context("Memory size parsing", func() {
		DescribeTable("should parse various memory size formats",
			func(input string, expected int64, shouldFail bool) {
				os.Setenv("GPU_DEVICE_0_ACTIVE_THREAD_PERCENTAGE", "10")
				os.Setenv("GPU_DEVICE_0_MEMORY_LIMIT", input)

				config := &common.Configuration{}
				calc, err := llmdinferencesim.NewResourceCalculator(config)

				if shouldFail {
					Expect(err).To(HaveOccurred())
				} else {
					Expect(err).NotTo(HaveOccurred())
					Expect(calc).NotTo(BeNil())
					limits := calc.GetGPULimits()
					Expect(limits[0].MemoryLimitBytes).To(Equal(expected))
				}
			},
			Entry("Bytes", "1024", int64(1024), false),
			Entry("Bytes with B", "1024B", int64(1024), false),
			Entry("Kilobytes", "10KB", int64(10000), false),
			Entry("Kibibytes", "10KiB", int64(10240), false),
			Entry("Megabytes", "100MB", int64(100000000), false),
			Entry("Mebibytes", "100MiB", int64(104857600), false),
			Entry("Gigabytes", "12GB", int64(12000000000), false),
			Entry("Gibibytes", "12Gi", int64(12884901888), false),
			Entry("Gibibytes alt", "12GiB", int64(12884901888), false),
			Entry("Terabytes", "1TB", int64(1000000000000), false),
			Entry("Tebibytes", "1TiB", int64(1099511627776), false),
			Entry("Float value", "1.5GB", int64(1500000000), false),
			Entry("Invalid unit", "10XB", int64(0), true),
			Entry("No number", "GB", int64(0), true),
			Entry("Empty", "", int64(0), true),
		)
	})
})

// Helper function to split environment variable string
func splitEnv(env string) (key, val string, ok bool) {
	for i := 0; i < len(env); i++ {
		if env[i] == '=' {
			return env[:i], env[i+1:], true
		}
	}
	return "", "", false
}
