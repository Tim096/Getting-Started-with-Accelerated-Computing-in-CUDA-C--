### GPU 計算與優化

1. 請解釋 GPU 的記憶體層級結構，以及如何有效利用這些不同層級的記憶體？  
2. 什麼是 memory coalescing？為什麼它對 GPU 性能很重要？  
3. 請說明 GPU warp/wavefront 的概念，以及如何避免 warp divergence？  
4. 比較 CUDA 和 HIP 的主要差異，如何將 CUDA 程式轉換為 HIP？  
5. 解釋 GPU kernel occupancy 的概念，以及如何優化它？  
6. 如何處理 GPU 中的 bank conflicts？  
7. ROCm 平台中，如何實現高效能的 kernel fusion？  
8. 請解釋 GPU Thread Hierarchy（Grid、Block、Thread）的概念，並說明其對演算法設計的影響。  
9. 為什麼需要使用 Shared Memory？它如何幫助提升效能？  
10. 請設計一個簡單的 GPU Kernel，並解釋如何分析它的效能瓶頸。  
11. 在 CUDA/HIP 編程中，如何有效管理 Host 和 Device 之間的記憶體傳輸？  
12. 如何使用 Profiling 工具（如 Nsight Compute、ROCm Profiler、rocprof）來分析 GPU kernel 的性能及識別瓶頸？  
13. 如何解決 GPU Kernel 中的競爭條件問題？  
14. 請解釋 Atomic Operation 的作用及其影響。  
15. 如何在 GPU Kernel 中使用組合語言或內建函式（Intrinsic Functions）提升效能？  
16. 請說明 Loop Unrolling 在 GPU Kernel 中的應用及其效能提升原理。  
17. 如何使用哪些工具來分析 GPU kernel 的性能（如 Nsight、rocprof）？  
18. 如何識別和解決 GPU 程式中的記憶體瓶頸？  
19. 如何評估和優化 kernel launch overhead？  
20. 描述常見的 GPU 性能反模式及解決方案。  
21. 如何進行 CPU-GPU 工作負載平衡？  
22. 解釋如何使用 hardware performance counters 進行性能分析。  
23. 使用 GPU 進行矩陣運算時，如何分配 thread blocks 並利用 Tiling 技術進行加速？  
24. 請實作一個簡單的矩陣乘法 Kernel 並解釋如何用 Shared Memory 提升效能。  
25. 當記憶體為 row-major 或 column-major 排列時，如何設計 GPU 演算法最佳化資料訪問模式？  
26. 若要在 GPU 上處理稀疏矩陣乘法，該如何實作？並請解釋 CSR 格式及相關演算法設計。

---

### 組合語言與底層優化

1. 如何使用向量化指令(AVX/SSE)優化計算密集的操作？  
2. 請解釋 CPU pipeline stall 的原因及如何避免？  
3. 如何使用 inline assembly 來確保特定的指令序列執行？  
4. 解釋 memory barrier 的概念和使用場景。  
5. 如何使用 prefetch 指令來優化記憶體存取？  
6. 描述如何手動展開迴圈(Loop Unrolling)來提升性能。  
7. 如何避免分支預測失敗造成的性能損失？  
8. 如何用組合語言實作矩陣乘法的核心迴圈？  
9. 在組合語言中，如何有效管理暫存器以最佳化運算？  
10. 為什麼有時需要手動撰寫組合語言來避免編譯器不佳的優化結果？  
11. 如何使用內建函式（如 `_mm256_add_ps`）加速 SIMD 運算？  
12. 請解釋指令流水線（Instruction Pipelining）的原理，並如何在程式中利用？  
13. 如何避免指令執行中的 Data Hazard？  
14. 如何用組合語言將 C 的 for 迴圈寫出並解釋執行流程？  
15. 用組合語言撰寫演算法的挑戰有哪些？如何克服？

---

### 並行計算與多執行緒

1. 如何實現一個 thread-safe 的記憶體池？  
2. 描述 false sharing 的問題及解決方案。  
3. 如何使用 atomic operations 來優化同步操作？  
4. 解釋 work stealing 演算法的實現方式。  
5. 如何設計一個高效能的 producer-consumer queue？  
6. 描述 OpenMP 中 schedule 子句的不同選項及使用場景。  
7. 如何處理多執行緒程式中的 deadlock 問題？  
8. 在多執行緒環境中，如何設計安全的資料共享？  
9. 請解釋 Mutex 和 Semaphore 的差別，並提供一個使用場景。  
10. 請說明 Work Stealing 是什麼，並如何應用於提升效能？  
11. 如何設計一個能平衡執行緒工作負載的演算法？  
12. 如何分析多執行緒程式的效能瓶頸（如 False Sharing）？  
13. 在記憶體階層的角度，如何減少 Cache Line 的競爭？  
14. 如何避免 race condition？  
15. 在多執行緒程式中如何進行同步？

---

### AI 操作優化 (ML Operator 與 Kernel 開發)

1. 如何優化卷積(Convolution)運算的性能？  
2. 描述矩陣乘法中的 Winograd 算法。  
3. 如何實現高效能的 batch normalization？  
4. 解釋 tensor core 的運作原理及如何利用它。  
5. 如何優化 ReLU、Sigmoid 等激活函數的計算？  
6. 描述 im2col 優化技巧及其適用場景。  
7. 如何處理稀疏矩陣運算的優化？  
8. 什麼是 ML Operator？請舉例並解釋在 Machine Learning 中的作用。  
9. 如何最佳化特定 ML Operator（如卷積）的效能？  
10. 請說明什麼是 Graphics Kernel，並舉例如何進行效能優化。  
11. 假設一個 Kernel 表現不好，你會如何排查效能問題？  
12. 在深度學習中，如何選擇合適的硬體加速（如 GPU、TPU）以達成最佳化？  
13. 如何處理不同硬體架構下的 Operator 相容性問題？

---

### 效能分析與調校

1. 如何使用 rocprof 進行性能分析？  
2. 如何識別和解決記憶體瓶頸？(已列於 GPU 分析區，但此為重複概念，已保留一次)  
3. 如何評估和優化 kernel launch overhead？(已列)  
4. 描述常見的 GPU 性能反模式及解決方案。(已列)  
5. 如何進行 CPU-GPU 工作負載平衡？(已列)  
6. 解釋如何使用 hardware performance counters。(已列)  
7. 如果程式執行效能不佳，如何進行效能分析與優化？

(以上有部分問題在前面已出現，此處不再重複列出)

---

### 底層系統知識

1. 解釋 cache coherency 協議的工作原理。  
2. 描述 virtual memory page fault 的處理流程。  
3. 如何優化 page table walk 的效能？  
4. 解釋 NUMA 架構的特點及其對程式設計的影響。  
5. 描述現代 CPU 中的 out-of-order execution。  
6. 如何處理和優化 DMA 傳輸？  
7. 畫出 CPU、Cache 和 Storage 之間的資料流，並解釋各層級的作用。  
8. 解釋 CPU Cache 的作用，以及如何透過演算法最佳化 Cache 命中率。

---

### 實作與特定演算法問題

1. 實現一個高效能的矩陣轉置函數。  
2. 設計一個 cache-oblivious 的排序算法。  
3. 實現一個支援 GPU 的記憶體池管理器。  
4. 設計一個高效能的 reduction kernel。  
5. 實現一個支援多 GPU 的矩陣乘法。  
6. 如何實現高效能的 parallel scan？  
7. 描述 parallel reduction 的優化技巧。  
8. 如何實現高效能的 parallel sort？  
9. 描述 parallel prefix sum 的實現方式。  
10. 給定 A(M×K) 與 B(K×N)，請寫程式完成矩陣相乘。  
11. 如何用多個 CPU 加速矩陣乘法？  
12. 若記憶體受限但需處理大矩陣運算，你會如何處理？  
13. 若採用多執行緒（Multithreading）加速矩陣乘法，你的設計策略是什麼？  
14. 用 OpenMP 展開矩陣乘法的迴圈該如何實作？  
15. 如何用 bit operation 將一個 byte 反轉？  
16. Merge Two Sorted Lists 的實作。  
17. N-ary Tree 的 Postorder Traversal 的實作。  
18. 如何使用 Tiling 技術優化程式效能？(已於 GPU 矩陣乘法最佳化提及，故不重複)

---

### 優化策略

1. 描述 memory bandwidth bound 與 compute bound 的區別及識別方法。  
2. 如何選擇最佳的 thread block size？  
3. 描述常見的 loop optimization 技術。（例如 Loop Unrolling、Blocking、Vectorization 等）  
4. 如何處理 GPU kernel 中的條件分支？  
5. 解釋 instruction level parallelism 的優化技巧。

(部分已和前面內容重疊，但仍為具體問法，故保留)

---

### 程式碼品質與測試

1. 如何確保優化後的程式碼正確性？  
2. 描述如何進行 GPU kernel 的單元測試。  
3. 如何處理浮點數計算的精度問題？  
4. 描述效能回歸測試的方法。

---

### 工具與平台

1. 你是否使用過 ROCm 平台？如何評估其效能？  
2. 與 CUDA 相比，ROCm 在功能上有何不同？優劣勢是什麼？

---

### 職位與經驗相關問題

1. 你對這個職位的理解是什麼？  
2. 你認為我們的 Team 主要在做什麼？  
3. 你如何看待 CUDA、CUBLAS 與其他加速 ML Operator 的競爭？  
4. 請分享你過去有關計算機架構、底層優化或演算法設計的經驗。  
5. 你是否修過計算機架構相關課程？學到了什麼？  
6. 曾經參與過什麼類似的專案嗎？  
7. 描述一個你曾經進行的效能優化專案，並說明你的具體貢獻。  
8. 在過去的專案中，你如何應對技術挑戰並提出創新的解決方案？

---

### 行為與個人問題

1. 請簡單介紹你自己。  
2. 在團隊中你通常扮演什麼角色？你的優缺點是什麼？  
3. 如果遇到技術問題無法解決，你會如何處理？  
4. 在過去的專案中，有哪些挑戰讓你學習到最多？  
5. 你如何看待遠端工作的挑戰和優勢？  
6. 你對早上 7:00 或晚上 12:00 的會議有什麼看法？