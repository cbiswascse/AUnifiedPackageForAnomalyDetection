<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
 

  <h3 align="center">Efficient Method for Optimizing Anomaly Detection with Clustering Algorithms<br> and for Unifiying in a Package</h3>

  <p>
    To create a common platform for anomaly detection process with some popular clustering algorithms to be an easy solution<br> for data analysis to verify the process data with other clustering algorithms.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<p>
The world of data is growing very fast, and it is a new challenge for data analysis to develop new methods to handle this massive <br> amount of data. A large number of data have many hidden factors that need to be identified and used for different algorithms. <br>Clustering is one of the significant parts of data mining. The term comes from the idea of classifying unsupervised data. Now-a-days <br>a lot of algorithms are implemented. Besides that, all those algorithms have some limitations, creating an opportunity to innovate new <br>algorithms for clustering. The clustering process can be separated in six different ways: partitioning, hierarchical, density, gridmodel,<br> and constraint-based models. The aim of the package is to implement various types of clustering algorithms and helps to deter-<br>mine which one is more accurate on detecting impure data from a large data set. To create a common platform for Some popluar  <br>algorithms for anomaly detection are implemented and converged all of them into a package(AnDe). The algorithms which are <br> implemented and combined into the package are: K-means, DBSCAN, HDBSCAN, Isolation Forest, Local Outlier Factor and<br> Agglomerative Hierarchical Clustering. The package reduce the consumption of time by compressing implementation hurdles of <br>each algorithms. The package is also makes the anomaly detection procedure more robust by visualizing in a more precise way <br>along with visualization of comparison in performance(accuracy, runtime and memory consumption) of those algorithm <br>implemented.

### Built With

For using this package, some popular packages are need to be configured in the working environment.

* [Numpy](https://numpy.org/)
* [Panda](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Time](https://docs.python.org/3/library/time.html)
* [os](https://docs.python.org/3/library/os.html)
* [Sklearn](https://scikit-learn.org/stable/)
* [Hdbscan](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
* [Tracemalloc](https://docs.python.domainunion.de/3/library/tracemalloc.html)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you set up thie pacage and use in you script.

### Prerequisites

At first, need install the package in your working environment for using this package. 
  ```sh
  pip install python=3.8
  ```
  ```sh
  pip install numpy
  ```
  ```sh
  pip install pandas
  ```
  ```sh
  pip install matplotlib
  ```
  ```sh
  pip install time
  ```
  ```sh
  pip install os
  ```
  ```sh
  pip install sklearn
  ```
```sh
  pip install Hdbscan
  ```
  ```sh
  pip install Tracemalloc
  ```
### Installation

1. Download the package from (https://github.com/cbiswascse/AUnifiedPackageForAnomalyDetection)
2. Install the package in you environment.
   ```sh
   pip install cb-cluster
   ```
3. Import the pacage in your script.
   ```sh
   from EMOADCAUP import Cluster
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

1. Call the cluster function.
	```sh
   from Ande import AnDe 
   AnDe.ClusterView()
   ```
2. Input the Location of CSV file.
	```sh
   Please, Input the Location of CSV:
   ```
3. Select yes(y) If you have Catagorical data in your dataset.
 	```sh
   Do you want to include Catagorical data [y/n]:
   ```
4. Select yes(y) If you want to scaling your dataset with MinMaxScaler.
	```sh
   Scaling data with MinMaxScaler [y/n]:
   ```
5. Available Clusering Algorithm
	Kmeans
	Dbscan 
	Isolation Forest 
	Local Factor Outlier 
	Hdbscan 
	Agglomerative
	```sh
   Choose your Algorithm:
   ```
####Kmeans Clusering
6. Number of Cluster
	```sh
   How many clusters you want?:
   ```
7. Select one of Average Method for Performance Metrics
```sh
   weighted,micro,macro,binary
   ```
####Dbscan
8. Input a Epsilon value
	```sh
   epsilon in Decimal:
   ```
9. Input a Min Samples value
```sh
   Min Samples In Integer:
   ```
10. Select one of Average Method for Performance Metrics
	```sh
   weighted,micro,macro,binary
   ```
####Hdbscan
11. Minimum size of cluster
	```sh
	Minimun size of clusters you want?:
   ```
12. Select one of Average Method for Performance Metrics
	```sh
   weighted,micro,macro,binary
   ```
####Isolation Forest
13. Contamination value
```sh
   Contamination value between [0,0.5]:
   ```
14. Select one of Average Method for Performance Metrics
	```sh
   weighted,micro,macro,binary
   ```
####Local Outlier Factor
13. Contamination value
```sh
   Contamination value between [0,0.5]:
   ```
14. Select one of Average Method for Performance Metrics
	```sh
   weighted,micro,macro,binary
   ```
#### Agglomerative

15. Number of Cluster
	```sh
   How many clusters you want?:
   ```
16. Select one of Average Method for Performance Metrics
```sh
   weighted,micro,macro,binary
   ```
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Chandrima Biswas - cbiswascse26@gmail.com

Project Link: [https://github.com/cbiswascse/AUnifiedPackageForAnomalyDetection](https://github.com/cbiswascse/AUnifiedPackageForAnomalyDetection)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
<p>
I would like to convey my heartfelt appreciation to my supervisor Prof.Dr. Doina Logofatu,for all her feedback, guidance, and <br>evaluations during the work. Without her unique ideas, as well as her unwavering support and  encouragement, I would <br>never have been able to complete this project.  In spite of her hectic schedule, she listened to my problem and gavethe appropriate advice. .Furthermore, I express my very profound gratitude<br> Prof. Dr. Peter Nauth for being the second supervisor of this work.