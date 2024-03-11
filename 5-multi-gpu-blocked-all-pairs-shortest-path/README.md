# Blocked All-Pairs Shortest Path (Multi-cards)

## Implementation

### How do you divide your data?

First, I add the additional data to make the length of data is divided by 64. For example, if the total metrics is 82, we will add the additional data to make its length become 64*2 = **128**.

I choose 64 as my blocking factors.

The configuration of my code is in Table.1:

| Kernel function | Grid Size             | Block Size |
| --------------- | --------------------- | ---------- |
| phase1          |     1                 | (32, 32)   |
| phase2          | (Round, 2)            | (32, 32)   |
| phase3          | (Round, Round)        | (32, 32)   |

> Table 1. The configuration in phase1,2,3

```c
//The width of the block is 64*64, and we divide into four 32*32 block.
// * means the data that might be executed in first iteration.
// Their index is (i,j), (i+32,j), (i,j+32), (i+32,j+32)
 ----- -----
| *   |*    |
|     |     |
 ----- -----
|*    |*    |
|     |     |
 ----- -----
```

### How do you implement the communication?

In phase 3, the image is divided into two regions: the red region is assigned to GPU0, and the blue region is assigned to GPU1. During block Floyd-Warshall computation, the only dependencies occur in the pivot-row and pivot-column areas. However, as shown in the diagram below, it is sufficient to transmit only the pivot-row. This is because the pivot-column has already been transmitted with the pivot-row in the previous iteration, and the values beyond the pivot-row will not be computed until phase 2 updates the pivot-column values.

```c
    *          // Modified
    *          // Modified
* * * * * * *  // Modified
    *          // Unmodified
    *          // Unmodified
    *          // Unmodified
    *          // Unmodified
```

I utilized OpenMP to generate two threads, each controlling a separate GPU. I transferred the pivot row between GPUs using cudaMemcpyPeer for inter-GPU communication.

![](https://i.imgur.com/cQaUpHK.png)

![](https://i.imgur.com/Nn3nXtm.png)

## Experiment & Analysis

### System Spec

hades.cs.nthu.edu.tw

### Weak Scalability

Observe weak scalability of the mulit-GPU implementations.

### Time Distribution

Analyze the time spent in:

1. computing
2. communication
3. memory copy (H2D, D2H)
4. I/O of your program w.r.t. input size.

|Testcase|v|Problem Size|Total(s)|Input(s)|Output(s)|Compute(s)|
|-|-|-|-|-|-|-|
|c01|5	    |    125	    |0.531330	|0.300358	|0.000056	|0.230916
|c02|160	|   4096000	    |0.468061	|0.255587	|0.000153	|0.212321
|c03|999	|  97002999	    |0.508910	|0.282370	|0.005140	|0.221400
|c04|5000	|1.25E+11	    |1.060658	|0.688902	|0.044676	|0.327080
|c05|11000	|1.331E+12	    |1.716411	|0.434437	|0.185830	|1.096144
|p16|16000	|4.096E+12	    |3.951392	|1.204183	|0.026570	|2.720639
|p21|20959	|9.20686E+12	|6.493136	|0.920829	|0.026806	|5.545501
|p26|25889	|1.73519E+13	|11.101869  |1.180990	|0.033938	|9.886941
|p31|31000	|2.9791E+13	    |20.570848  |2.400624	|1.380707	|16.789517
|p34|34921	|4.25853E+13	|26.761774  |2.764130	|0.045377	|23.952267
|c06|39857	|6.33161E+13	|38.578996  |2.847176	|2.268196	|33.463624
|c07|44939	|9.07549E+13	|53.879104  |3.154683	|2.881724	|47.842697

> ![Total Time](https://i.imgur.com/H6DQqDk.png)
> Fig. 1 Total time when problem size grows.
 
In Fig.1, it can be observed that as the problem size increases, the time also increases almost proportionally, indicating good scalability of the program.

> ![GPU Time Distribution](https://i.imgur.com/9lToRDD.png)
> Fig.2 GPU Time Distribution

Fig.2 illustrates that the computation time increases with the number of nodes, and we can also see that communication time, which initially did not occupy much time, becomes a significant portion of the total time as the size increases.

> ![GPU Time Distribution 2](https://i.imgur.com/vCk1tZz.png)
> Fig.3 GPU Time Distribution Percentage

Fig.3 provides an overview of the percentage distribution of each component's time. As the size increases, the proportion of communication time tends to increase, while computation time decreases slightly. This suggests that when dealing with a large number of vertices, optimization or reduction of communication time between multiple GPUs should be considered.

## Experience & conclusion

This experiment taught me how to utilize multiple GPUs for computation. It's crucial to adjust the algorithm appropriately; otherwise, the dual-GPU speed might even be slower than single GPU, with most of the time spent on unnecessary communication.

## Images

![](https://i.imgur.com/rySFMik.png)

![](https://i.imgur.com/6QGInp7.png)

![](https://i.imgur.com/MBEa5id.png)

![](https://i.imgur.com/CdmyLmv.png)

![](https://i.imgur.com/XydFEJb.png)

![](https://i.imgur.com/x8ej7aK.png)

![](https://i.imgur.com/3fjeyfr.png)

![](https://i.imgur.com/7N4e2fM.png)

![](https://i.imgur.com/Kr7ChVu.png)

![](https://i.imgur.com/RgGFxUk.png)

![](https://i.imgur.com/Nh9dM6f.png)

![](https://i.imgur.com/KtEVF9N.png)

![](https://i.imgur.com/Bk4LidX.png)



