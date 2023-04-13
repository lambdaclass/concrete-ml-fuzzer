# concrete-ml-fuzzer
A differential fuzzer to compare [concrete-ml](https://docs.zama.ai/concrete-ml) against [scikit-learn](https://scikit-learn.org)

The fuzzer is configured with certain pre-defined values.

1. To ensure a high precision, a large number of bits is set, typically 11 bits. 
2. The project claims to achieve a high accuracy of 99% by using a large number of bits for quantization, therefore, we use this value as a reference to determine the correctness of the results obtained by comparing them with the expected output."
