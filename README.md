# ULRec

1.  Important packages

<!---->

    python                    3.7.0
    higher                    0.2.1              
    pytorch-gpu               1.9.0   
    scikit-learn              0.23.2
    scipy                     1.5.3

&#x20;2\. Run the codes:&#x20;

    cd unlearnable-examples-<dataset name>
    python main.py -c 0 -k 1 -g 0

c\=0 denotes the process of data attacks. c\=1/2/3 denotes evaluating the attack with BPR, GCN, and ItemCF, respectively.&#x20;

k\=1/2/3 denotes that the number of privacy-concerned users is 50/100/500.&#x20;

g denotes the id of gpu.&#x20;
