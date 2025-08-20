# FedECA using Flower

This repo replicates FedECA results as shown in the original [FedECA quickstart example](https://github.com/owkin/fedeca/blob/main/quickstart/quickstart.md) (Distributed Analysis section). For this purpose, we have copied the example CSV data files provided by the `FedECA` repository and we put them under `data/`.

The Flower FedECA implementation can be run both in simulation and deployment. By default it will run without bootstrapping and it will report both `FedECA (naïve)` and `FedECA (robust)`, both after running 10 steps of NewtonRaphson. You can enable bootstrapping by toggling the `bootstrap` setting in `pyproject.toml` to true. Both `global` (default) and `per-client` bootstrapping are supported.

## Setup the Environment

On a new environment with Python 3.10-3.12 do, install the app:

```shell
# cd to this directory, then
pip install -e .
```

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
> For more insights on how to use Flower's Deployment Engine, check the [documentation](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html). You may refer also to the [how to run Flower with Docker](https://flower.ai/docs/framework/docker/index.html) guide.

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

With the infrastructure (i.e. `SuperLink` and `SuperNodes`) ready, you can submit a run. You can achieve so by pointing the `flwr run` command to another federation (in this case a real one created with the compose file). If you inspect the `pyproject.toml` you'll find this section at the bottom:

```TOML
[tool.flwr.federations.remote]
address = "127.0.0.1:9093"
insecure = true
```

Where `address` indicates the address of the `SuperLink`. Here for testing purposes we don't use TLS certificates (hence the `insecure = true`). In a new terminal, submit the run:

```shell
# "remote" is the name in the federation (you can change it)
# "--stream" enables you to see the logs generated by the ServerApp
# on the terminal you run `flwr run` from.
flwr run . remote --stream
```
