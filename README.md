# Classification of random data into Probability Distributions

## Assignment

The assignment for the semester project was to create a program that can classify input data into one of four distributions: **Normal/Gaussian**, **Poisson**,
**Exponential**, or **Uniform**. The program must output values characterizing the distribution and justify its result at the end of the computation. The
assignment also defines the following limits that must be adhered to:
* The test file will be several GB in size
* Memory will be limited to 1GB
* The program must finish within 15 minutes on an iCore7 Skylake.
  
The program must accept the following input parameters:
* **file** - path to the file, can be relative to program.exe or absolute
* **processor** - strings specifying which processors the computation will run on simultaneously
  * all - uses CPU and all available GPUs
  * SMP - multithreaded computation on CPU
  * names of OpenCL devices as separate arguments - note that there may be several OpenCL platforms in the system

The program also includes a watchdog thread that monitors the correct
functioning of the program. 
I extended the input arguments to include specifying the number of threads to be used on each CPU, selecting an optimised run, and selecting the frequency at which the watchdog must check the results.


## User documentation
### Controls
Mandatory arguments must be entered to run the program:
* The path to the data file
* Program mode
  * smp
  * all
  * The names of the individual OpenCL devices

In addition to the mandatory program input arguments, I have added the following optional arguments:
* w is an integer argument and affects the sleep time of the Watchdog, specified in seconds.
  
Example of a valid input that will turn on the program in SMP mode without optimization and the watchdog thread will check the program run every 6 seconds:

```
"C:\gauss" smp -w 5
```

### Output
After processing the file, the program will always output the following information:

* Input parameters with which the program was run
* The computed statistics
* RSS results
* Run time
* Resulting distribution with parameters calculated based on the lowest RSS value

#### Example of program output in SMP mode with input file (7 GB) containing data with normal distribution:

<details close>
<summary>Expand example</summary>

```bash
                        [Initial parameters]
-------------------------------------------------
> File:                         C:\delete\gauss
> Mode:                         smp
> Number of threads:            8
> Optimalization:               TRUE
> Watchdog timer:               2 sec


> Started ..

                        [Statistics]
-------------------------------------------------
> n:                            1000000000
> sum:                          -3572.2
> mean:                         -3.5722e-06
> variance:                     1.00005
> min:                          -6.10922
> max:                          5.87261
> isNegative:                   1
> isInteger:                    0


                        [Results]
-------------------------------------------------
> Gauss RSS:                    0.00079459
> Poisson RSS:                  -nan(ind)
> Exponential RSS:              0.588353
> Uniform RSS:                  0.477286



                        [Time]
-------------------------------------------------
> Statistics computing time:    3.02456 sec.
> Histogram computing time:     4.37617 sec.
> RSS computing time:           0.0003251 sec.
> TOTAL TIME:                   7.40135 sec.

> Input data have 'Gauss/Normal distribution' with mean=-3.5722e-06 and variance=1.00005
```

</details>

#### Example of program output in SMP mode with input file (1 GB) containing data with exponential distribution:

<details close>
<summary>Expand example</summary>
  
```bash
                        [Initial parameters]
-------------------------------------------------
> File:                         C:\delete\exp
> Mode:                         smp
> Number of threads:            8
> Optimalization:               TRUE
> Watchdog timer:               2 sec


> Started ..

                        [Statistics]
-------------------------------------------------
> n:                            137524224
> sum:                          1.37528e+08
> mean:                         1.00003
> variance:                     1.00015
> min:                          0
> max:                          20.9085
> isNegative:                   0
> isInteger:                    0


                        [Results]
-------------------------------------------------
> Gauss RSS:                    0.24959
> Poisson RSS:                  0.17973
> Exponential RSS:              0.121527
> Uniform RSS:                  0.558112


                        [Time]
-------------------------------------------------
> Statistics computing time:    4.54918 sec.
> Histogram computing time:     0.803888 sec.
> RSS computing time:           0.0002501 sec.
> TOTAL TIME:                   5.35362 sec.

> Input data have 'Exponential distribution' with lambda=0.99997
```

</details>

#### Example of program output in SMP mode with an input file (1 GB) containing data with a Poisson distribution:

<details close>
<summary>Expand example</summary>

```bash
                        [Initial parameters]
-------------------------------------------------
> File:                         C:\delete\poisson
> Mode:                         smp
> Number of threads:            8
> Optimalization:               TRUE
> Watchdog timer:               2 sec


> Started ..

                        [Statistics]
-------------------------------------------------
> n:                            141318656
> sum:                          1.41323e+08
> mean:                         1.00003
> variance:                     0.999858
> min:                          0
> max:                          10
> isNegative:                   0
> isInteger:                    1


                        [Results]
-------------------------------------------------
> Gauss RSS:                    0.0203551
> Poisson RSS:                  2.58775e-09
> Exponential RSS:              0.402114
> Uniform RSS:                  0.208506


                        [Time]
-------------------------------------------------
> Statistics computing time:    3.32517 sec.
> Histogram computing time:     0.688783 sec.
> RSS computing time:           0.0001357 sec.
> TOTAL TIME:                   4.01441 sec.

> Input data have 'Poisson distribution' with lambda=1.00003

```

</details>

#### Example of program output in SMP mode with an input file (1 GB) containing data with a uniform distribution:

<details close>
<summary>Expand example</summary>
  
```bash
                        [Initial parameters]
-------------------------------------------------
> File:                         C:\delete\uniform
> Mode:                         smp
> Number of threads:            8
> Optimalization:               TRUE
> Watchdog timer:               2 sec


> Started ..

                        [Statistics]
-------------------------------------------------
> n:                            133684224
> sum:                          6.68395e+07
> mean:                         0.49998
> variance:                     0.0833297
> min:                          0
> max:                          1
> isNegative:                   0
> isInteger:                    0


                        [Results]
-------------------------------------------------
> Gauss RSS:                    3.82184
> Poisson RSS:                  7.97942
> Exponential RSS:              7.39103
> Uniform RSS:                  0.0357194


                        [Time]
-------------------------------------------------
> Statistics computing time:    4.0244 sec.
> Histogram computing time:     0.654257 sec.
> RSS computing time:           0.0001364 sec.
> TOTAL TIME:                   4.67911 sec.

> Input data have 'Uniform distribution' with a=0 and b=1
```

</details>

### Analyzing Results

Table 5.1 and Figure 5.1 show the speedup of each program mode compared to the sequential mode. We were able to achieve significant computation speedup in SMP mode using auto-vectorization and also in ALL mode using dynamic resource allocation to threads.
The only slow mode (as opposed to sequential computation) was the computation using only "Intel(R) UHD Graphics 620", which is due to the lowest computational power of the device compared to the others.

Table 5.1: Comparison of the speeds of each program mode versus sequential mode (16 GB file size, SSD)

Table 5.2: SMP-vectorization. Sample time of each part of the program. 16 GB, SSD disk)

Table 5.3: ALL mode. 16 GB, SSD disk)

Figure 5.1: Comparison of the speed of each program mode (size of the co
16 GB, SSD)

### Conclusion
The assignment for the term paper has been completed. The program provides an accurate probability estimate for the input data, based on the calculated RSS (residual sum of squares) value. 
The program was tested on a Windows 11 operating system with an Intel Core i7 processor and an NVIDIA GeForce MX150 graphics card, with the data located on an SSD disk. 
The majority of my time was dedicated to the development of an algorithmic approach for the accurate identification of the optimal distribution. The subsequent challenge was to ascertain why the maximum CPU utilisation could not be employed. This work provided a practical overview of the functioning of threads and memory pages. 
The most significant slowdowns in the program are attributable to the absence of pages in RAM. Furthermore, the speed of the program in SMP mode is influenced by the type of disk on which the data resides. Despite efforts, parallelisation using OpenCL devices has not resulted in faster processing. The time and memory required for data transfer to the device proved to be a significant limitation. The most efficient mode of the program was identified as SMP mode with automatic vectorisation.
All individual data and their comparisons, including comparisons of computation speeds, are available in an Excel spreadsheet in the folder with this documentation.
