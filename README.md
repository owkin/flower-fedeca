# FedECA

This repository contains the official implementation of the [FedECA method](https://doi.org/10.1038/s41467-025-62525-z) in the [Flower framework](https://flower.ai/).

The original implementation in the [Substra framework](https://docs.substra.org/en/stable/) used in the FedECA article is available [here](https://github.com/owkin/fedeca).

Please [cite our paper](#citing-fedeca) if you use our code!


## Setup the Environment

On a new environment with Python 3.10-3.12, install the app:

```shell
# cd to this directory, then
pip install -e .
```


## Toy model

For illustration purpose, a toy example of "distributed datasets" can be found in the `data` folder. The example contains four data centers, each holding a synthetic dataset of right-censored time-to-event data with 10 covariates. All demonstrations below are based on this toy example.

FedECA runs in both simulation and deployment modes. By default, it runs without bootstrapping and reports results with `naïve` and `robust` variance estimates, both after 10 Newton-Raphson steps. Bootstrapping can be enabled by setting the `bootstrap` parameter in `pyproject.toml` to `true`. Both `global` (default) and `per-client` bootstrapping are supported.


## Run in simulation

From this directory, run the app:

```shell
flwr run .
```

You will see a log like the following:

```shell
Loading project configuration...
Success
INFO :      (iter: 0) Success after 5 FedCox steps (norm_delta = 6.527887117250224e-09)
-------------- FedECA (naïve) ---------------
       coef  se(coef)  coef lower 95%  coef upper 95%         z         p  exp(coef)  exp(coef) lower 95%  exp(coef) upper 95%
0  0.041718  0.049127        -0.05457        0.138005  0.849178  0.395782     1.0426             0.946892             1.147981
-------------- FedECA (robust) ---------------
       coef  se(coef)  coef lower 95%  coef upper 95%         z         p  exp(coef)  exp(coef) lower 95%  exp(coef) upper 95%
0  0.041718  0.070581       -0.096618        0.180054  0.591062  0.554479     1.0426             0.907902             1.197282
```

You can extend the above command to modify the config used (see the `[tool.flwr.app.config]` section in the `pyproject.toml`). Let's say you want to do 20 steps of Federated NewtonRaphson. Then start the run like this:

```shell
flwr run . --run-config="nr-iterations=20"
```

And if you want to enable `bootstrapping` and run for 200 iterations (default is `20`), do:

```shell
flwr run . --run-config="bootstrap=true bootstrap-iterations=200"

# Will report a summary:
-------------- Bootstrap FedECA (n_iter = 200 + 1) ---------------
       coef  se(coef)  coef lower 95%  coef upper 95%         z         p  exp(coef)  exp(coef) lower 95%  exp(coef) upper 95%
0  0.041718  0.039048       -0.034816        0.118251  1.068359  0.285359     1.0426             0.965783             1.125527
```

By default `"global"` bootstrapping is used but you can enable per-client `bootstrapping` by overriding `bootstrap-fn` like this:

```shell
flwr run . --run-config="bootstrap=true bootstrap-iterations=200 bootstrap-fn='per-client'"

# Will report a summary:
-------------- Bootstrap FedECA (n_iter = 200 + 1) ---------------
       coef  se(coef)  coef lower 95%  coef upper 95%        z         p  exp(coef)  exp(coef) lower 95%  exp(coef) upper 95%
0  0.041718  0.037539       -0.031857        0.115292  1.11132  0.266431     1.0426             0.968645             1.122202
```


## Run in deployment

> \[!TIP\]
> For more insights on how to use Flower's Deployment Engine, including setting up TLS and authentication, check the [documentation](https://flower.ai/docs/framework/deploy.html). You may refer also to the [how to run Flower with Docker](https://flower.ai/docs/framework/docker/index.html) guide.

You can spawn a Flower federation comprised of one `SuperLink` and four `SuperNodes` by running the `compose.yaml` file in this repository. You'll need to first build the image described in the `Dockerfile`. This will (1) install the dependencies in the `pyproject.toml` and (2) copy the content of the `data/` directory to the containers.

```shell
# Build the container (only needed the first time or if you edit Dockerfile)
# This image will be used by both SuperLink and all SuperNodes
docker build . -t flower-fedeca-base-with-deps

# There after, spawn the federation
docker compose up
```

On a new terminal, you may use `docker stats` to see the services that have been launched. You'll see something like:

```shell
CONTAINER ID   NAME                                CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O    PIDS
69d56719c413   owkin-flower-fedeca-supernode-1-1   1.17%     48.14MiB / 31.29GiB   0.15%     11.2kB / 7.88kB   0B / 4.1kB   33
25617260c2f7   owkin-flower-fedeca-supernode-4-1   1.29%     48.16MiB / 31.29GiB   0.15%     11.2kB / 7.94kB   0B / 4.1kB   33
e628e5cd63f9   owkin-flower-fedeca-supernode-3-1   1.07%     48.14MiB / 31.29GiB   0.15%     11.2kB / 8.01kB   0B / 4.1kB   33
9239157488ab   owkin-flower-fedeca-supernode-2-1   1.23%     48.14MiB / 31.29GiB   0.15%     11.1kB / 8.07kB   0B / 4.1kB   33
204f2b4e4a0c   owkin-flower-fedeca-superlink-1     0.49%     48.5MiB / 31.29GiB    0.15%     30.6kB / 11.7kB   0B / 4.1kB   38
```

With the infrastructure (i.e. `SuperLink` and `SuperNodes`) ready, you can submit a run. You can achieve so by pointing the `flwr run` command to another federation (in this case a real one created with the compose file). If you inspect the `pyproject.toml` you'll find the `remote` federation at the bottom. Run the Flower App in your deployed federation with:

```shell
# "remote" is the name in the federation (you can change it)
# "--stream" enables you to see the logs generated by the ServerApp
# on the terminal you run `flwr run` from.
flwr run . remote --stream
```

## Bringing your own data

The above demonstrations use the [toy example data](#toy-model) pre-partitioned into four centers. When the FedECA app runs, in simulation or deployment mode, the `ClientApp`s load the appropriate datasets by following the data path _template_ defined in the `[tool.flwr.app.config]` section of the `pyproject.toml` as:

```TOML
[tool.flwr.app.config]
...
path-to-data = "data/center{}/data.csv"
```
Then, at runtime, each `ClientApp` completes that path with their respective partition id. 

To test FedECA with your own data (currently only `csv` files are supported):
1. Partition your data on a per-center basis into different `csv` files. For examle, name them `partition1.csv`, `partition2.csv`, etc.
2. In the `pyproject.toml`, set `path-to-data = "path/to/my/partition{}.csv"`. Alternatively, you can override this setting at runtime as:
    ```shell
       flwr run . --run-config="path-to-data='path/to/my/partition{}.csv'"
    ```
3. Optionally, if run with the Simulation Engine, make sure the `options.num-supernodes` in `pyproject.toml` sets the appropriate number of supernodes equal to the number of data partitions. If run with the Deployment Engine, make sure the `compose.yaml` file is spawning enough `SuperNodes`.


## Citing FedECA

```bibtex
@article{OgierduTerrail2025,
  author = {Jean Ogier du Terrail and Quentin Klopfenstein and Honghao Li and Imke Mayer and Nicolas Loiseau and Mohammad Hallal and Michael Debouver and Thibault Camalon and Thibault Fouqueray and Jorge Arellano Castro and Zahia Yanes and Laëtitia Dahan and Julien Taïeb and Pierre Laurent-Puig and Jean-Baptiste Bachet and Shulin Zhao and Remy Nicolle and Jérôme Cros and Daniel Gonzalez and Robert Carreras-Torres and Adelaida Garcia Velasco and Kawther Abdilleh and Sudheer Doss and Félix Balazard and Mathieu Andreux},
  title = {FedECA: federated external control arms for causal inference with time-to-event data in distributed settings},
  journal = {Nature Communications},
  year = {2025},
  volume = {16},
  number = {1},
  pages = {7496},
  doi = {10.1038/s41467-025-62525-z},
  url = {https://doi.org/10.1038/s41467-025-62525-z},
  abstract = {External control arms can inform early clinical development of experimental drugs and provide efficacy evidence for regulatory approval. However, accessing sufficient real-world or historical clinical trials data is challenging. Indeed, regulations protecting patients’ rights by strictly controlling data processing make pooling data from multiple sources in a central server often difficult. To address these limitations, we develop a method that leverages federated learning to enable inverse probability of treatment weighting for time-to-event outcomes on separate cohorts without needing to pool data. To showcase its potential, we apply it in different settings of increasing complexity, culminating with a real-world use-case in which our method is used to compare the treatment effect of two approved chemotherapy regimens using data from three separate cohorts of patients with metastatic pancreatic cancer. By sharing our code, we hope it will foster the creation of federated research networks and thus accelerate drug development.},
  issn = {2041-1723}
}
```
