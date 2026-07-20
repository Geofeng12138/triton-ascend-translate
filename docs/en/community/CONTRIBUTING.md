# Contribution Guide


- [Getting Started](#入门.md)
- [Developer Certificate of Origin (DCO)](#开发者来源认证.md)
- [Development Guide](#开发指导.md)
  - [Code Style](#代码风格.md)
  - [Fork-Pull Development Model](#Fork-Pull开发模式.md)
  - [Code Gate Exception Handling](#代码门禁异常处理.md)
  - [ISSUE Guidelines](#ISSUE规范.md)
  - [Submitting a PR](#提出PR.md)


<h2 id="入门.md">Getting Started</h2>


- Fork the Triton-Ascend repository on [GitHub](https://github.com/triton-lang/triton-ascend).
- Read [README.md](https://github.com/triton-lang/triton-ascend/blob/main/README.md) for project information and setting up the development environment.


<h2 id="developer-certificate-of-origin.md">Developer Certificate of Origin (DCO)</h2>


All submissions must include a `Signed-off-by:` line, automatically added using `git commit -s`.


```bash
git commit -s -m "your commit message"
```


This will automatically add a line `Signed-off-by: Your Name <your.email@example.com>` at the end of the commit message, indicating that you confirm the origin and authorization of the contribution.


<h2 id="开发指导.md">Development Guide</h2>


- **[Code Style](#code-style.md)**
- **[Fork-Pull Development Mode](#fork-pull-development-mode.md)**
- **[Code Gate Exception Handling](#code-gate-exception-handling.md)**
- **[ISSUE Guidelines](#issue-guidelines.md)**
- **[Submitting a PR](#submitting-a-pr.md)**


<h2 id="code-style.md">Code Style</h2>


Please follow the coding style below to make Triton Ascend easy to develop, maintain, and review.


- Coding Guidelines


Please use the unified coding style of the Triton Ascend community. The recommended coding style for Python is [PEP 8 Coding Style](https://pep8.org/), and the recommended style for C++ is [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). You can use [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy), [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net/), [CMakeLint](https://github.com/cmake-lint/cmake-lint), [CodeSpell](https://github.com/codespell-project/codespell), [ShellCheck](https://github.com/koalaman/shellcheck), and [pylint](https://pylint.org/) to check code formatting. It is recommended to install these plugins in your IDE.


- Unit Testing Guide


Please use the unified unit testing style of the Triton Ascend community. The recommended unit testing style for Python is [pytest](http://www.pytest.org/en/latest/), and for C++ it is [Googletest Primer](https://github.com/google/googletest/blob/main/docs/primer.md). The design intent of a test case should be reflected through its comment name. For test case design, please refer to the [gather test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/custom_op/test_gather_load.py) and the [layer_norm test case](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/05-layer-norm.py).


- Refactoring Guide


We encourage developers to refactor our code to eliminate [code smells]. Refactored code should also follow the coding style and testing style requirements. When you receive a warning, you need to refactor the code to be merged.


<h2 id="Fork-Pull开发模式.md">Fork-Pull Development Model</h2>


1. Fork the Triton Ascend project


Before submitting your code to the Triton Ascend project, please ensure that you have forked the Triton Ascend project to your own repository. Subsequently, you will develop on your forked project and merge it into the Triton Ascend project via a Pull Request. This means there is parallel development between the Triton Ascend repository and your own repository, so please ensure consistency between the repositories.


2. Clone the remote repository


Clone your forked Triton Ascend project using git and add the upstream repository:


```shell
git clone https://github.com/{your_forked_repo}/triton-ascend.git && cd triton-ascend && git submodule update --init --depth 1
git remote add upstream https://github.com/triton-lang/triton-ascend.git
```


3. Develop code in a local environment


Before developing your code, you need to set up the development environment according to the [Triton Ascend Installation Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/installation_guide.md).


To avoid inconsistency issues between multiple branches, please create a new local development branch for developing new features.


```shell
git checkout -b {new_branch_name} origin/main
git fetch upstream       # Fetch the latest code from the upstream repository
git rebase upstream/main # Rebase onto the latest upstream trunk
```


Taking the main branch as an example, Triton Ascend may create version branches or downstream development branches as needed. Once you have created your branch and synchronized updates from the upstream main branch, you can start developing your code.


4. Self-testing of code changes


After completing the code changes, please verify that your changes pass the tests.


Write test case code for your developed code under the `ascend/examples/pytest_ut` path of the local code branch, and verify your test script in the local environment to ensure that your changes can pass the tests.


5. Push the code to the remote repository


After completing code updates & testing, push your commit to your remote repository.


```shell
git add .
git status #Check the updated files
git commit -s -m "your commit message"
git push origin {your_new_branch_name}
```


6. Create a pull request to the Triton Ascend main repository.


After pushing the code to your remote repository, you need to create a Pull Request between your new branch and the Triton Ascend main branch. Once the new merge request is created, "Jenkins CI" will be automatically set up to run your pipeline tests. Please merge your Pull Request into the upstream main branch as soon as possible to reduce merge risks.


<h2 id="code-gate-exception-handling.md">Code Gate Exception Handling</h2>


Code gate access anomalies mainly include the following situations. Please resolve the gate access anomaly issues based on the relevant prompt information.


- Compilation failed


Please check the cause of the compilation failure based on the prompt information, resolve it, and then recompile.


- Static check failed


Please identify the exception information in the code based on the prompt and resolve it.


- CI pipeline did not pass


Please check the prompt information to identify the test cases that failed in the CI pipeline, investigate the causes, resolve the issues, and then re-run the CI pipeline.


<h2 id="ISSUE规范.md">ISSUE Guidelines</h2>


A good way to contribute to the project is to send detailed reports when encountering issues. We always greatly appreciate well-written, thorough bug reports and will be very grateful to you for them!


When reporting issues, please refer to the following format:


- What software versions are used in your environment (Triton Ascend, Python, OS, etc.)?
- Is this a bug report or a feature request?
- What kind of issue are you reporting? Add the corresponding labels to highlight it on the issue dashboard.
- What happened?
- What did you expect to happen?
- How can it be reproduced? (Be as precise as possible)


You can also choose one of the predefined [issue templates](https://github.com/triton-lang/triton-ascend/issues/new/choose).


Issue Consultation:


- If you find an open issue that is exactly what you want to solve, please comment on that issue to let others know you will be working on it.
- If the issue has been open for a while, please perform a pre-check before resolving it.
- If you solve an issue you reported yourself, you still need to let others know before closing the issue.


<h2 id="submit-a-pr.md">Submit a PR</h2>


- Submit your ideas as issues on [GitHub](https://github.com/triton-lang/triton-ascend).
- If the new feature to be developed requires extensive design details, you should also submit a design proposal.
- After reaching consensus through issue discussion and design proposal review, proceed with forking the repository for development and submitting a PR.
- No PR is allowed until it receives 2+ LGTM (Looks Good To Me) from Approvers. Note that reviewers are not permitted to add LGTM to their own PRs.
- After sufficient discussion, the PR will be merged, rejected, or abandoned based on the outcome of the discussion.


## Notes


- Avoid any unrelated changes.
- Ensure your commit history is clean and organized.
- Before creating a PR, rebase with the latest code from the upstream repository.
- For bug fix PRs, make sure to link all related Issues and PRs.

