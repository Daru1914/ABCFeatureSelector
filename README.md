# ABCFeatureSelector

Optimization framework for feature selection in Python source code quality assessment. Using nature-inspired Modified Discrete Artificial Bee Colony (MDisABC) algorithm and meta-analysis, it efficiently reduces dimensionality while preserving important relationships. The project applies optimization techniques to select minimal yet effective sets of code metrics.

## Classical ABC diagram

![classical_abc_workflow](https://github.com/user-attachments/assets/807ecb12-d524-4135-a4b7-55cbddef3701)

## MDisABC diagram

![mdisabc_workflow](https://github.com/user-attachments/assets/72fbf0e9-6398-431f-a650-01f104f5acef)

## Code Metrics

| Metric ID |         Metric Full Name           | Metric Description |
|-----------|------------------------------------|------------|
| HDIF      |        Halstead Difficulty         | The difficulty of understanding or maintaining a piece of code. It takes into account the number of distinct operators and operands, as well as their occurrences, to estimate the cognitive effort required to comprehend the code. Higher Halstead Difficulty values suggest more challenging code that may be harder to maintain or troubleshoot. |
| NUMPAR    |        Number of Parameters        | Number of the parameters of the method/function. The varargs and kwargs parameters counts as one. |
| NOI       |   Number of Outgoing Invocations   | Number of directly called methods/functions from a specific method. If a method/function is invoked several times, it is counted only once. |
| AP        |            Anti Pattern            |      .      |
| HPV       |     Halstead Program Vocabulary    | Total number of distinct operators and operands used in a code segment. It quantifies the variety of unique symbols (operators and operands) employed in the code. A higher Halstead Program Vocabulary indicates a code segment with a more diverse set of language elements. |
| CompMR    |      Complexity Metric Rules       |      .      |
| TCLOC     |     Total Comment Lines of Code    | Number of comment and documentation code lines of the method/function, including its local classes. |
| TLOC      |         Total Lines of Code        | Number of code lines of the method/function, including empty and comment lines, as well as its nested functions. |
| CLOC      |        Comment Lines of Code       | Number of comment and documentation code lines of the method/function, excluding its local classes. |
| DLOC      |     Documentation Lines of Code    | Number of documentation code lines of the method/function. |
| TNOS      |     Total Number of Statements     | Number of statements in the method/function, including the statements of its nested functions. |
| NOS       |        Number of Statements        | Number of statements in the method/function, excluding the statements of its nested functions. |
| TLLOC     |    Total Logical Lines of Code     | Number of non-empty and non-comment code lines of the method/function, including the non-empty and non-comment lines of its nested functions. |
| LOC       |           Lines of Code            | Number of code lines of the method/function, including empty and comment lines; however, its nested functions are not included. |
| LLOC      |       Logical Lines of Code        | Number of non-empty and non-comment code lines of the method/function, excluding its nested functions. |
| SMR       |         Size Metric Rules          |      .      |
| McCC      |   McCabe’s Cyclomatic Complexity   | Complexity of the method/function expressed as the number of independent control flow paths in it. It represents a lower bound for the number of possible execution paths in the source code and at the same time it is an upper bound for the minimum number of test cases needed for achieving full branch test coverage. The value of the metric is calculated as the number of the following instructions plus 1: if, for, while, except and conditional expression. Moreover, logical “and” and logical “or” expressions also add 1 to the value because their short-circuit evaluation can cause branching depending on the first operand. The following instructions are not included: else, try, finally. |
| WI        |            Warning Info            |            |
| HPL       |      Halstead Program Length       | The total length of a program based on the number of operators and operands used. It provides a measure of the program’s length in terms of vocabulary and complexity. A higher Halstead Program Length indicates a more extensive program in terms of code elements. |
| HCPL      | Halstead Calculated Program Length | The expected length of a program in terms of the number of operators and operands used in the code. It is calculated based on the Halstead software science metrics, which consider the number of distinct operators and operands, as well as their occurrences. A higher HCPL indicates a more complex program in terms of the vocabulary and operators used. |
| HVOL      |          Halstead Volume           | Combines Halstead Length and Program Vocabulary to estimate the size or volume of a code segment. It provides a measure of the overall size of the code in terms of operators and operands. A higher Halstead Volume indicates a larger and potentiall more complex code segment. |
| HNDB      |  Halstead Number of Delivered Bugs | Number of potential defects (bugs) in a code segment. It uses Halstead Volume and Difficulty to predict the number of bugs that might be present in the code. A higher number of delivered bugs suggests a code segment that is more likely to contain defects. |
| HTRP      |  Halstead Time Required to Program | The time it would take to write or program a code segment based on the Halstead metrics. It uses Halstead Length, Volume, and Difficulty to predict the time required for implementation. This metric provides an estimate of the effort or time needed to develop the code. |
| HEFF      |          Halstead Effort           | Combines Halstead Length and Difficulty to estimate the total effort required to understand, implement, and maintain a piece of code. It quantifies the cognitive complexity and implementation effort associated with a code segment. Higher Halstead Effort values indicate greater effort required to work with the code. |
| CoupMR    |        Coupling Metric Rules       |      .      |
| CD        |          Comment Density           | Ratio of the comment lines of the method/function (CLOC) to the sum of its comment (CLOC) and logical lines of code (LLOC). |
| NL        |           Nesting Level            | Сomplexity of the method/function expressed as the depth of the maximum embeddedness of its conditional, iteration and exception handling block scopes. The following instructions are taken into account: if, else-if, else, for, while, with, try, except, finally. |
| NLE       |       Nesting Level Else-If        | Сomplexity of the method/function expressed as the depth of the maximum embeddedness of its conditional, iteration and exception handling block scopes, where in the if-else-if construct only the first if instruction is considered. The following instructions are taken into account: if, else, for, while, with, try, except, finally. The following instructions do not increase the value by themselves; however, if additional embeddedness can be found in their blocks, they are considered: else-if (i.e. in the if-else-if construct the use of else-if does not increase the value of the metric). |
| NII       |   Number of Incoming Invocations   | Number of other methods/functions and attribute initializations which directly call the method/function. If the method/function is invoked several times from the same method/function or attribute initialization, it is counted only once. |
| TCD       |       Total Comment Density        | Ratio of the total comment lines of the method/function (TCLOC) to the sum of its total comment (TCLOC) and total logical lines of code (TLLOC). |
| DMR       |     Documentation Metric Rules     |      .      |
| WC        |          Warning Critical          |            |
| WMaj      |           Warning Major            |            |
| WB        |          Warning Blocker           |            |
| WMin      |           Warning Minor            |            |