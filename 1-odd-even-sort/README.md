# Odd Even Sort

## Implementation

### How do you handle an arbitrary number of input items and processes?

My idea is as follows:

1. If the number of processes is greater than the length of the array, each process will be assigned one until all are assigned.
   Example: If there are 5 processes and the array length is 3, the allocation will be (1,1,1,0,0).

2. First, distribute the array evenly among the processes. In the last round, distribute the remaining elements one by one until all are assigned.
   Example: If there are 5 processes and the array length is 13, first calculate `13/5 = 2`, allocate 2 elements to each process, and then distribute the remaining `13%5=3` elements one by one to each process until all are assigned. So, it will become (3,3,3,2,2).

### How do you sort in your program?

My approach is as follows:

1. Sort each process individually using the spreadsort algorithm from the Boost library.
2. When comparing data between Process A and Process B, start by comparing the end of the data in Process A with the beginning of the data in Process B. If the condition is met, it means the maximum number in Process A is smaller than the minimum number in Process B, indicating that sorting is already completed. Process A will then send a signal to Process B indicating that sorting is done. If the condition is not met, merge and sort the data from both processes simultaneously, and then send the length of the larger portion back to Process B.

### Steps


#### Part 1. Initialization

```c
    if(argc != 4){
		fprintf(stderr, "Must provide 3 parameters\n");
		return -1;
	}

	MPI_Init(&argc,&argv);

	unsigned int array_length = atoll(argv[1]);
	char *input_filename = argv[2];
	char *output_filename = argv[3]; 
	
	int rank, total_process;	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the rank (id) of the calling process
	MPI_Comm_size(MPI_COMM_WORLD, &total_process); // the total number of processes

	MPI_File f;
	MPI_File f2;
	MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f2);

	unsigned int data_length = calculateDataLength(total_process, &array_length, rank);

	int max_rank;
	if(total_process > array_length) max_rank = array_length - 1;
	else max_rank = total_process - 1;
```

- First, we check if the length of the parameters is 4. If it's not 4, it means there was an input error by the user, and the program exits.
- `MPI_INIT()` initializes MPI.
- The type of `array_length` is set to `unsigned int` because the specification states that the range of the array length value is `1 ≤ n ≤ 536870911`, which is approximately a quarter of `INT_MAX`. Here, we use `unsigned int` for potentially faster performance.
- `MPI_File_open()` is used with three parameters: MPI_MODE_RDONLY (read), MPI_MODE_CREATE (create file), and MPI_MODE_WONLY (write).
- Here, a custom function `calculateDataLength` is used to calculate the length of data that each rank needs to process. `calculateDataLength` calculates the length of data each process should be allocated based on the process's rank, array length, and number of processes. The method is similar to the one described above, aiming to distribute data as evenly as possible among all processes.

```c
unsigned int calculateDataLength(int total_process, unsigned int *array_length, int rank) {
	// array length:  5
	// total process: 13
	if(total_process >= *array_length) {
		if(rank < *array_length) return 1;
		else return 0;
	}
	// array length: 13
	// total process: 5
	else {
		unsigned int rest = *array_length % total_process;
		unsigned int data_length = *array_length / total_process;
		if(rank < rest) return data_length + 1;
		else return data_length;
	}
}
```

- Calculate the maximum rank index and store it in the variable `max_rank`.

#### Part 2: Reading Data and Pre-sorting

If the process's data length is not 0, proceed with the following steps:

- `(float*)data` stores the data allocated to this process.
- `(float*)new_data` stores the data received from other processes by this process.
- `(float*)tmp_data` stores temporary data when merging data from two processes.
- Calculate the start index for reading:
    - If rank <= rr, indicating the preceding processes allocated an extra array length, set `start_idx` as `(dd+1)*rank`.
    - If rank > rr, indicating processes without an extra allocation, set `start_idx` as `(dd+1) * rr + (rank-rr) * dd`.
- Use `MPI_File_read_at()` to read data of length `data length` starting from the specified `start index`. Since we have already calculated the positions from which each process should read, each process reads its allocated portion of data.
- Utilize `boost::spreadsort` to sort the data allocated to each process. Pre-sorting facilitates simpler sorting checks later on.

```c
if(data_length !=0){

    rr = array_length % total_process;
    dd = array_length / total_process;
    ddd = dd + 1;
    data = (float*)malloc(sizeof(float) * ddd);
    new_data = (float*)malloc(sizeof(float) * ddd);
    tmp_data = (float*)malloc(sizeof(float) * (ddd+dd));

    if(rank <= rr) start_idx = ddd * rank;
    else start_idx = ddd * rr + (rank - rr) * dd;

    MPI_File_read_at(f, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);

    boost::sort::spreadsort::spreadsort(data, data + data_length);
}
```

#### Part 3： If only one process is allowed, directly write the sorted result into the file and terminate.

```c
if(total_process == 1) {
    MPI_File_write_at(f2, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);
    free(data);
    MPI_Finalize();
    return 0;
}
```

#### Part 4： ODD-EVEN Sort

- This section mainly involves a while loop to perform the ODD-EVEN Sort. We define a variable `sum_of_sorted` to represent the total number of processes that have been sorted.
- Upon entering the loop, we initialize the variable `sorted` to 1, assuming that the data is already sorted. If later it turns out that the data is not sorted, we set `sorted` to 0.
- We only need to perform computations if `data_length != 0`. When `data_length = 0`, it means that the process has no data to process, so `sorted` is directly set to 1.
- If the rank is odd:
    - Send our own data to the process with an even rank using `MPI_Isend()`. `MPI_Isend()` is a non-blocking API.
- If the rank is even and not the last process:
    - Calculate the length of the received data and store it in the variable `new_data_length`.
    - Use `MPI_Recv()` to receive the data.
- If the maximum value in our data is greater than the minimum value of the received data, it means that the data is not sorted yet. In this case, we perform the following steps:
    - Set `sorted` to 0 to indicate that the data is not sorted yet.
    - Use `mergeArray()` to merge and sort the data from both sides into `tmp_data`.
    - Use a for loop to copy the sorted data from `tmp_data` back to `data`, starting from index 0 with a length of `data_length`.
    - Use `MPI_Isend()` to send a flag to inform other processes that sorting has been performed again.
    - Use `MPI_Isend()` to send the processed data to other processes.
- If the maximum value in our data is less than the minimum value of the received data, it means that the data is already sorted. In this case, we send a flag to inform other processes that sorting is not needed, and we can use the original data.
- If the rank is odd, wait for the flag's return. If the flag is 1, indicating that sorting has been performed again, use `MPI_Recv()` to receive the data after re-sorting.
- The EVEN SORT section is similar to the ODD SORT section, with the main difference being the determination of odd and even nodes.
- Finally, use `MPI_Allreduce()` to calculate the sum of the variable `sorted` in each process using `MPI_SUM`. If the sum equals the total number of processes, it means that all processes have been sorted, and we can exit the while loop. If not, continue the while loop.
- After exiting the while loop, check if `data_length` is zero. If it is zero, it means there is no data to write; if not zero, write the data allocated to us according to the original `start_idx` into the FILE, free the allocated memory, and execute `MPI_FINALIZE()` to end the MPI program.

```c
while (sum_of_sorted < total_process) {
    sorted = 1;
    if (data_length != 0) {
        if (!isEven) {
            MPI_Isend(data, data_length, MPI_FLOAT, rank - 1, TAG1, MPI_COMM_WORLD, & request1);
        }
        if (isEven && rank != max_rank) {
            unsigned int new_data_length = calculateDataLength(total_process, & array_length, rank + 1);
            MPI_Recv(new_data, new_data_length, MPI_FLOAT, rank + 1, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // if not sorted
            // merge two sort lists and return max one
            //
            if (data[data_length - 1] > new_data[0]) {
                sorted = 0;
                unsigned int tmp_data_length = data_length + new_data_length;
                // merge two array and sort
                mergeArray(data, new_data, & data_length, & new_data_length, tmp_data);
                for (int i = 0; i < data_length; ++i) {
                    data[i] = tmp_data[i];
                }
                flag = 1;
                MPI_Isend( & flag, 1, MPI_INT, rank + 1, TAG5, MPI_COMM_WORLD, & request1);
                MPI_Isend( & tmp_data[data_length], new_data_length, MPI_FLOAT, rank + 1, TAG2, MPI_COMM_WORLD, & request2);
            }
            // if sorted
            // send nothing to rank + 1
            else {
                flag = 0;
                MPI_Isend( & flag, 1, MPI_INT, rank + 1, TAG5, MPI_COMM_WORLD, & request1);
            }
        }
        if (!isEven) {
            MPI_Recv( & flag_buf, 1, MPI_INT, rank - 1, TAG5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (flag_buf == 1) {
                MPI_Recv(data, data_length, MPI_FLOAT, rank - 1, TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        // EVEN SORT
        //..........
        //..........
         
    }
    MPI_Allreduce( & sorted, & sum_of_sorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

if (data_length != 0) {

    MPI_File_write_at(f2, sizeof(float) * start_idx, data, data_length, MPI_FLOAT, MPI_STATUS_IGNORE);

    free(data);
    free(new_data);
    free(tmp_data);
}

MPI_Finalize();

return 0;
```



## Experiment & Analysis 

### Methodology

#### System Spec (If you run your experiments on your own cluster)

apollo.cs.nthu.edu.tw

#### Performance Metrics: How do you measure the computing time, communication time and IO time? How do you compute the values in the plots?

I use `MPI_Wtime()` to wrap the code before and after the calculations. For example, `MPI_File_open` is an IO operation, so we add its time to the variable `io_time`. 

We declare variables `comm_time` and `io_time` at the beginning to store different times.

```c
double t1, t2, t3, t4, t5, t6, t7, t8;
double comm_time = 0;
double io_time = 0;

MPI_Init();

t1 = MPI_Wtime();
MPI_File f;
MPI_File f2;
MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f2);
t2 = MPI_Wtime();
io_time += (t2-t1);
```

![](https://i.imgur.com/PmS7aIi.png)

We submit tasks using Sbatch and generate Slurm files to analyze logs. The log files contain the execution time of each process. We take the average of these times as the total time. Subtracting `comm_time` and `io_time` from the total time will give us an approximation of the CPU computation time.

1-1.sh
```bash
#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
echo "N1 n1"
srun time ./hw1 536869888 /home/pp20/share/hw1/testcases/35.in ./out
```

1-2.sh
```bash
#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
echo "N1 n2"
srun time ./hw1 536869888 /home/pp20/share/hw1/testcases/35.in ./out
```

test.sh
```bash
sbatch 1-1.sh
sbatch 1-2.sh
```

```bash
sh test.sh
```

### Plots: Speedup Factor & Time Profile

#### Scenario 1: Fixed Node=1, Different Numbers of Processes

Table 1 lists the performance comparison of different numbers of processes when the node count is fixed at 1. From Fig. 1, we can observe that CPU Time decreases as the number of processes increases, while IO Time remains relatively constant. COMM Time, on the other hand, increases with the number of processes because more processes require inter-process communication. Additionally, although the COMM TIME for process=8 is less than that for process=4, when viewed as a percentage in Fig. 3, the proportion of COMM TIME increases as the number of processes increases, confirming our hypothesis.

|# of Nodes | # of process | Comm. Time | IO Time | CPU Time | Total Time |Speedup|
|-----------|--------------|------------|---------|----------|------------|-------|
|1|1|	0.00|	4.46|	26.18	|30.64|	1.00|
|1|2|	2.79|	3.41|	14.88	|21.08|	1.45|
|1|4|	5.92|	2.84|	10.20	|18.96|	1.62|
|1|8|	5.26|	2.36|	6.44	|14.05|	2.18|


Table 1. CPU, COMM, IO Time Under different number of process when node=1.

![](https://i.imgur.com/gtUiKY0.png)


Fig. 1 CPU, COMM, IO Time under different number of process when node=1.

![](https://i.imgur.com/3RcRFTw.png)

Fig. 2 Speedup under different number of process when node=1.

![](https://i.imgur.com/KXP2iD7.png)


Fig. 3 Each stage time percentage when node=1.


#### Scenario 2: Fixed Number of Processes, Different Numbers of Nodes

When the number of processes is fixed at 12, we observe that the total time decreases as the number of nodes increases. This suggests that when distributed across multiple nodes, each process receives more computational resources, leading to improved performance.

|# of Nodes | # of process | Comm. Time | IO Time | CPU Time | Total Time |Speedup|
|-----------|--------------|------------|---------|----------|------------|-------|
|1|12|6.70|2.21|6.71|15.61|1.00|
|2|12|6.51|1.97|6.69|15.17|1.03|
|3|12|6.02|2.03|6.53|14.58|1.07|
|4|12|5.43|2.09|6.39|13.91|1.12|

Table.2 Time Consumption when total process=12

![](https://i.imgur.com/q1Sqzej.png)

Fig.3 Time Distribution Chart when total processes = 12.

![](https://i.imgur.com/jfoEFmu.png)

Fig.4 Speed-up Line Chart when total processes = 12.

#### Scenario 3: Different Number of Processes with Fixed Nodes

We fixed the number of nodes at 4 and observed whether the number of processes would affect the time. Here, we found that CPU time basically decreases as the number of processes increases, but at the same time, we also noticed that the proportion of COMM time becomes increasingly higher.

|# of Nodes | # of process | Comm. Time | IO Time | CPU Time | Total Time |Speedup|
|-----------|--------------|------------|---------|----------|------------|-------|
|4|4	|5.58	|3.40	|8.99	|17.97|1.00|
|4|8	|5.55	|2.22	|6.96	|14.73|1.22|
|4|12	|5.66	|2.03	|6.26	|13.95|1.29|
|4|16	|6.69	|1.88	|5.98	|14.55|1.24|
|4|20	|6.58	|3.93	|5.83	|16.34|1.10|
|4|24	|6.37	|2.08	|5.00	|13.44|1.34|
|4|28	|6.33	|2.02	|5.13	|13.48|1.33|
|4|32	|6.56	|2.13	|4.63	|13.32|1.35|
|4|36	|6.54	|3.29	|5.40	|15.22|1.18|
|4|40	|6.64	|4.26	|5.14	|16.04|1.12|
|4|44	|6.45	|2.42	|4.23	|13.11|1.37|
|4|48	|6.79	|2.72	|4.54	|14.05|1.28|



![](https://i.imgur.com/tYd6Sy4.png)

![](https://i.imgur.com/5QvBRq1.png)

![](https://i.imgur.com/yPxMVpQ.png)


#### Part 4: Different sorting function

At first, I used `std::sort()` because I remembered that `std::sort` is fast. However, after researching online and consulting with classmates, I heard about a boost library that can significantly speed up sorting. After using it, my program became much faster. Therefore, I conducted a small experiment to compare the speed of different sorting algorithms.

|Sorting Algorithm| Comm. Time | IO Time | CPU Time | Total Time |
|-|-|-|-|-|-|
|boost::spread_sort	|5.42|1.96| 6.22|13.61|
|boost::pdqsort	    |5.40|1.92| 6.51|13.82|
|std::sort	        |5.41|1.88| 9.11|16.40|
|std::stable_sort	|5.44|1.88|10.00|17.32|

From the charts, we can observe that `spread_sort` is the fastest among all sorting algorithms. We can see that faster sorting algorithms can save a significant amount of CPU time, leading to a noticeable decrease in overall speed.

![Different algo.](https://i.imgur.com/1MFunVF.png)



### Discussion (Must base on the results in your plots)

#### Compare I/O, CPU, Network performance. Which is/are the bottleneck(s)? Why? How could it be improved?

When the number of processes is small, the CPU becomes the primary bottleneck because there is little need for message passing. As the number of processes increases, the workload is distributed across many CPUs for computation, leading to a gradual decrease in CPU time until it reaches a bottleneck where further reductions are challenging. At this point, network performance becomes the main bottleneck because communication between different processes requires network transmission, which affects the speed of communication. However, reducing the number of processes to decrease COMM Time will also impact computation time. Therefore, it's necessary to find the optimal number of processes based on the distribution pattern of the data.



#### Compare scalability. Does your program scale well? Why or why not? How can you achieve better scalability? You may discuss for the two implementations separately or together.

I feel that the scalability of my program is not very good. Increasing the number of processes did not significantly reduce the overall time. However, it's also possible that the number of experiments conducted was not sufficient, leading to errors. Nevertheless, the CPU Time of the program did decrease with an increase in the number of processes, indicating successful distributed computation.

### Others 

## Experiences / Conclusion
In conclusion, increasing the number of processes does not necessarily result in faster execution of the program. When there are too many processes, significant time is spent on data transmission and reception. Additionally, employing efficient sorting algorithms can greatly improve program performance.

## What have you learned from this assignment?
This assignment has deepened my understanding of MPI's working principles and how to run programs on multiple processes.

## What difficulties did you encounter in this assignment?
I spent a lot of time optimizing the code, but the reduction in execution time was minimal. Additionally, I initially encountered difficulties with MPI_IRecv() not receiving packets, but switching to MPI_Recv() resolved the issue.

## If you have any feedback, please write it here. Such as comments for improving the spec of this assignment, etc.
I hope that in the future, teaching assistants can provide tips and tricks for speeding up code execution. I am eager to learn how to accelerate programs effectively.
