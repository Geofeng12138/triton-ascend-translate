# Roadmap Guide


Use GitHub Issues to track and manage the plans and medium-to-long-term goals of each organization. This document provides references and guidelines for writing roadmap-type Issues in community projects, helping to create and maintain high-quality roadmaps.


Below is a complete roadmap Issue example, demonstrating the practical application of all recommended elements. It is recommended to first review this example to get an overall impression, then read the subsequent detailed specification explanations.


```markdown
创建 Issue 标题：[Roadmap] Triton-Ascend Roadmap 2026 Q2

---
# Triton-Ascend Roadmap 2026 Q2

This quarter focuses on upstream Triton compatibility alignment, Ascend NPU backend performance optimization, and operator coverage expansion, continuously improving compiler stability and ecosystem integration capability.

## Focus

- Upstream Compatibility: Align with upstream Triton compiler frontend and IR changes, reduce fork divergence
- Backend Performance: Optimize Ascend NPU kernel generation and memory scheduling strategy
- Operator Coverage: Expand supported Triton ops and data types, improve end-to-end model coverage
- Usability: Improve debugging tools and error reporting for Ascend backend

## Upstream Compatibility

- [ ] **Triton 3.x IR and frontend alignment**
Goal: Align forked Triton frontend with upstream Triton 3.x IR changes, reduce merge conflict and divergence
Issue: [Related Issue link]

- [ ] **Triton Language feature parity check**
Goal: Systematically check and supplement missing Triton language features (e.g., tl.dot with new dtypes, constexpr enhancements) on Ascend backend
Issue: [Related Issue link]

## Backend Performance

- [ ] **Ascend NPU kernel auto-tuning support**
Goal: Support Triton autotune mechanism on Ascend backend, enable dynamic kernel configuration selection
Owner: @contributor-a
Issue: [Related Issue link]

- [ ] **Memory scheduling and L2 cache optimization**
Goal: Optimize memory allocation strategy and L2 cache utilization in generated Ascend kernels for large-scale training scenarios
Issue: [Related Issue link]

## Operator Coverage

- [ ] **FP8 dtype and mixed-precision ops support [🙋 Help Wanted]**
Goal: Support FP8 (E4M3/E5M2) dtype in tl.dot and related ops on Ascend backend, enabling FP8 training workflows
Owner: TBD
Issue: [Related Issue link]

- [ ] **Additional Triton built-in ops support**
Goal: Add missing built-in ops (e.g., tl.cumsum, tl.reduce with custom axis, advanced indexing) on Ascend backend
Issue: [Related Issue link]

## Usability

- [ ] **Ascend backend debugging and error reporting improvement**
Goal: Improve Ascend backend error messages, add device-side debug print and kernel profiling support
Issue: [Related Issue link]

## Sub-issues

[Triton-Ascend Roadmap 2026 Q1 #xxx](link)
[FP8 Support Phase 2 #xxx](link)
```


## 1. Title Format


**Format:** `[Roadmap] <Project Name> Roadmap <Time Range>`, use Q1/Q2/Q3/Q4 for quarterly releases, and H1/H2 for semi-annual releases.


**Example:**


- `[Roadmap] Triton-Ascend Roadmap 2026 Q2`
- `[Roadmap] Triton-Ascend Roadmap 2026 H1`


## 2. Top-Level Content


### 2.1 Opening Description (Optional)


Provide a brief summary of the project overview, vision, or overall direction. For example, briefly describe the goals of Triton-Ascend for this quarter in terms of upstream alignment, backend performance, and operator coverage.


### 2.2 Key Focus Areas Section


List the 3-5 most critical focus areas for this cycle, recommended to be grouped by the project's **functional domains** or **technical modules**, covering a global perspective:


```markdown
## Focus

• Upstream Compatibility: Align with upstream Triton compiler frontend and IR changes, reduce fork divergence
• Backend Performance: Optimize Ascend NPU kernel generation and memory scheduling strategy
• Operator Coverage: Expand supported Triton ops and data types, improve end-to-end model coverage
• Usability: Improve debugging tools and error reporting for Ascend backend
• Ecosystem Integration: Enhance integration with PyTorch, vLLM, and MindSpeed on Ascend backend
```


**Features:**


- High-level summary and description of the main development directions for the current cycle project, without going into detailed specifics.


## 3. Main Functional Modules Section


### 3.1 Principles for Section Division


Group by the project's **functional areas** or **technical modules**, for example:


- **Upstream Compatibility** - Upstream Triton frontend and IR alignment
- **Backend Performance** - Ascend NPU backend kernel generation and memory optimization
- **Operator Coverage** - Triton built-in ops and dtype support on Ascend
- **Usability** - Debugging tools and error reporting
- **Ecosystem Integration** - Integration with training/inference frameworks


### 3.2 Module Structures


Each module contains multiple **specific work items** in the following format:


```markdown
## [Module Name]

- [ ] **Work item name/feature description**
Goal: [Goal description]
Owner: @GitHubID      [optional]
Issue: [Related Issue link]   [optional]
PR: [Related PR link]         [optional]

- [ ] **Another work item**
Goal: [Goal description]
Owner: @GitHubID      [optional]
Issue: [Related Issue link]   [optional]
PR: [Related PR link]         [optional]
```


## 4. Key Metadata Fields


Each work item should contain the following key information:


### 4.1 Goal


- **Meaning**: Work objective or brief description
- **Usage**: Describes the goal of the work item
- **Example**: `Goal: Support FP8 (E4M3/E5M2) data types in the Ascend backend tl.dot`


### 4.2 Owner


- **Meaning**: Responsible person  
- **Format**: `负责人：@GitHubID`  
- **Usage**: Specify who is responsible for or leads the work item  
- **Example**: `负责人：@contributor-a`


### 4.3 Issue


- **Meaning**: Associated GitHub Issue
- **Format**: `Issue: <Issue link>`
- **Usage**: Track detailed design and discussion
- **Example**: `Issue: https://github.com/triton-lang/triton-ascend/issues`


### 4.4 PR (Pull Request)


- **Meaning**: Related implementation PR
- **Format**: `PR：<PR link>`
- **Usage**: Link to implementation work
- **Example**: `PR：https://github.com/triton-lang/triton-ascend/pulls`


## 5. Optional Supplementary Content


### 5.1 🙋 Welcome Contribution Mark


For work items that especially welcome contributions from community developers, it is recommended to use the **[🙋 Welcome Contributions]** label to identify them.


```markdown
- [ ] **FP8 dtype and mixed-precision ops support [🙋 Help Wanted]**
Goal: Support FP8 (E4M3/E5M2) dtype in tl.dot and related ops on Ascend backend
Owner: TBD
Issue: #123
```


### 5.2 Sub-Issue


List cross-cycle related roadmap Issues or decomposition Issues of large work items at the bottom of the roadmap Issue.


**Difference from the Issue field in work items:**


- **Issue Field in Work Item**: Links to the specific design, discussion, or tracking issue for that work item  
- **Sub-Issue Section**: Used to associate roadmap issues from other cycles (such as unfinished work from the previous quarter) or to break down large work items into multiple independently tracked sub-issues


```markdown
## Sub-issues

[Triton-Ascend Roadmap 2026 Q1 #xxx](link)  <!-- Related previous quarter Roadmap -->
[FP8 Support Phase 2 #xxx](link)            <!-- Breakdown of a large work item -->
```

