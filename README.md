# Relay-FL
This is the simulation code package for the following paper:

Zehong Lin, Hang Liu, and Ying-Jun Angela Zhang. “Relay-Assisted Cooperative Federated Learning” to appear at IEEE Transactions on Wireless Communications, 2022. [[ArXiv Version](https://arxiv.org/abs/2107.09518)]

The package, written on Python 3 and Matlab, reproduces the numerical results of the proposed algorithm in the above paper.


## Abstract of Article:

> Federated learning (FL) has recently emerged as a promising technology to enable artificial intelligence (AI) at the network edge, where distributed mobile devices collaboratively train a shared AI model under the coordination of an edge server. To significantly improve the communication efficiency of FL, over-the-air computation allows a large number of mobile devices to concurrently upload their local models by exploiting the superposition property of wireless multi-access channels. Due to wireless channel fading, the model aggregation error at the edge server is dominated by the weakest channel among all devices, causing severe straggler issues. In this paper, we propose a relay-assisted cooperative FL scheme to effectively address the straggler issue. In particular, we deploy multiple half-duplex relays to cooperatively assist the devices in uploading the local model updates to the edge server. The nature of the over-the-air computation poses system objectives and constraints that are distinct from those in traditional relay communication systems. Moreover, the strong coupling between the design variables renders the optimization of such a system challenging. To tackle the issue, we propose an alternating-optimization-based algorithm to optimize the transceiver and relay operation with low complexity. Then, we analyze the model aggregation error in a single-relay case and show that our relay-assisted scheme achieves a smaller error than the one without relays provided that the relay transmit power and the relay channel gains are sufficiently large. The analysis provides critical insights on relay deployment in the implementation of cooperative FL. Extensive numerical results show that our design achieves faster convergence compared with state-of-the-art schemes.


## Dependencies
This package is written on Python 3. It requires the following libraries:
* Python >= 3.5
* torch
* torchvision
* scipy
* CUDA (if GPU is used)


## How to Use
The main file is **main.py**. It can take the following user-input parameters by a parser (also see the function **initial()** in main.py):

| Parameter Name  | Meaning| Default Value| Type/Range |
| ---------- | -----------|-----------|-----------|
| K   | total number of devices   |20   |int   |
| N   | total number of relays   |1   |int   |
| PL   | path loss exponent   |3.0   |float   |
| trial   | total number of Monte Carlo trials   |50   |int   |
| SNR   | -noise variance in dB   |100   |float   |
| P_r   | relay transmit power budget   |0.1   |float   |
| verbose   | output no/importatnt/detailed messages in running the scripts   |0   |0, 1   |
| seed   | random seed   |1   |int   |
| gpu  | GPU index used for learning (if possible)   |1   |int   |
| local_ep   | number of local epochs: E   |1   |int   |
| local_bs   | local batch size: B, 0 for full batch  |0   |int   |
| lr   | learning rate, lambda  |0.05  |float   |
| low_lr   | learning rate, bar_lambda  |1e-5  |float   |
| gamma   | learning rate decrease ratio, gamma  |0.9   |float   |
| step   | learning rate decrease step, bar_T  |50   |int   |
| momentum   | SGD momentum, used only for multiple local updates   |0.99   |float   |
| epochs   | number of training rounds, T   |500   |int   |
| iid   | 1 for iid, 0 for non-iid   |1   |0, 1   |
| noniid_level   | number of classes at each device for non-iid  |2   |2, 4, 6, 8, 10   |
| V_idx   | Variable index   |0  |int   |


Here is an example for executing the scripts in a Linux terminal:
> python -u main.py --gpu=0 --trial=50 --V_idx 0
