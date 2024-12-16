#include <stdio.h>

int main()
{
  /*
   * Device ID is required first to query the device.
   * 裝置ID是必須的第一步來查詢裝置
   */

  int deviceId;
  cudaGetDevice(&deviceId); // 裝置ID

  cudaDeviceProp props; // 裝置屬性
  cudaGetDeviceProperties(&props, deviceId); // 取得裝置屬性
  /*
   * `props` now contains several properties about the current device.
   * `props` 現在包含關於目前裝置的許多屬性
   */

  int computeCapabilityMajor = props.major;
  int computeCapabilityMinor = props.minor;
  int multiProcessorCount = props.multiProcessorCount;
  int warpSize = props.warpSize;

  // Print the device ID, number of SMs, compute capability major, compute capability minor, and warp size
  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
