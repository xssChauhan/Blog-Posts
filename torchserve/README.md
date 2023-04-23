# TorchServe Tutorial
- [TorchServe Tutorial](#torchserve-tutorial)
  - [Goals](#goals)
  - [Setup](#setup)
    - [1. Clone this repo](#1-clone-this-repo)
    - [2. Setup env and libraries](#2-setup-env-and-libraries)
    - [3. Install `TorchServe`](#3-install-torchserve)
    - [4. Download Transformers Model](#4-download-transformers-model)
    - [5. Package the model for torchserve / Download prepackaged model](#5-package-the-model-for-torchserve--download-prepackaged-model)
  - [Running and Inference](#running-and-inference)
    - [Start Docker Container](#start-docker-container)
    - [Running Inferences](#running-inferences)
  - [Parallelism Benefits](#parallelism-benefits)
    - [1 worker, 1 Request at a time](#1-worker-1-request-at-a-time)
    - [More workers?](#more-workers)
      - [2 Workers](#2-workers)
      - [4 workers](#4-workers)
  - [Conclusion](#conclusion)

## Goals
The goals of this tutorial are:
- Serve a model using `TorchServe` and `Docker`.
- Explore `TorchServe`'s inbuilt parallelism and compare performance.
- [ ] WIP: Add benchmarks for `TorchScript`. 

## Setup

### 1. Clone this repo
```
$ git clone https://github.com/xssChauhan/Blog-Posts.git
$ cd Blog-Posts/torchserve
```
### 2. Setup env and libraries

```bash
mamba create -n torchserve python=3.9 transformers=4.24.0

```

### 3. Install `TorchServe`

Follow the instructions [here](https://github.com/pytorch/serve/blob/master/README.md#install-torchserve).

### 4. Download Transformers Model

Clone the model from huggingface using:
```bash
$ git -c http.sslVerify=false clone https://huggingface.co/bert-base-uncased
```

It is important that the full files(not only the lfs pointers) for `config.json`, `vocab.txt` and `pytorch_model.bin` have been downloaded. You can also download the files directly from [here](https://huggingface.co/bert-base-uncased/tree/main).

### 5. Package the model for torchserve / Download prepackaged model

```bash
$ torch-model-archiver --model-name "bert" --version 1.0 --serialized-file bert-base-uncased/pytorch_model.bin --extra-files "bert-base-uncased/config.json, bert-base-uncased/vocab.txt" --handler bert_torch_serve_handler.py
```

Make a model store dir and move the packaged model in there:
```shell
$ mkdir model_store
$ mv bert.mar model_store/
```

## Running and Inference

### Start Docker Container

```bash
$ docker run -e "OMP_NUM_THREADS=1" -e "MKL_NUM_THREADS=1" --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -v $(pwd)/model_store:/home/model-server/model-store pytorch/torchserve:latest-cpu
```

By default a `TorchServe` container starts with no models active, so we use the in-built management API to start the model(Could take a few seconds to a minute to run):
```bash
$ curl -X POST "http://localhost:8081/models?model_name=bert&url=bert.mar&initial_workers=1"
```
This command spins up 1 worker for the model. 

Now we are ready to make calls to the container to run inferences.

### Running Inferences
Let's verify if the service has spun up correctly:

```bash
$ curl -X POST http://127.0.0.1:8080/predictions/bert -T input.txt
>> 1
```
Since the model has been randomly initialized, the output can be 0/1, but what's important is that an output was received.

## Parallelism Benefits

### 1 worker, 1 Request at a time

Since we started the container with 1 worker, we can directly use it, or run the following to make sure only 1 worker is running:
```bash
$ curl -v -X PUT "http://localhost:8081/models/bert?min_worker=1&max_worker=1&synchronous=true"
```

Let's run a benchmark of 10 inferences first sequentially, and then parallely:

Sequentially:
```bash
$ time seq 10  | xargs -n 1 -P 1 bash -c 'url="http://127.0.0.1:8080/predictions/bert"; curl -X POST $url -T input.txt'

>> 0.05s user 0.08s system 0% cpu 37.771 total
```

Parallelly:
```shell
$ time seq 10  | xargs -n 1 -P 2 bash -c 'url="http://127.0.0.1:8080/predictions/bert"; curl -X POST $url -T input.txt'

>> 0.00s user 0.00s system 59% cpu 38.201 total 
```
Obviously, since there are no parallel workers, we do not see any performance gain. Slight decrease could be attributed to the queue being used.

### More workers?

#### 2 Workers
Let's scale up to 2 workers:
```shell
$ curl -v -X PUT "http://localhost:8081/models/bert?min_worker=2&max_worker=2&synchronous=true"

*   Trying 127.0.0.1:8081...
* Connected to localhost (127.0.0.1) port 8081 (#0)
> PUT /models/bert?min_worker=2&max_worker=2&synchronous=true HTTP/1.1
> Host: localhost:8081
> User-Agent: curl/7.87.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: 258b2906-3110-47e0-8ad9-f4b9c27a81cf
< Pragma: no-cache
< Cache-Control: no-cache; no-store, must-revalidate, private
< Expires: Thu, 01 Jan 1970 00:00:00 UTC
< content-length: 54
< connection: keep-alive
< 
{
  "status": "Workers scaled to 2 for model: bert"
}
* Connection #0 to host localhost left intact
```

Sequentially:
```bash
$ time seq 10  | xargs -n 1 -P 1 bash -c 'url="http://127.0.0.1:8080/predictions/bert"; curl -X POST $url -T input.txt'

0.05s user 0.09s system 0% cpu 38.274 total
```

Parallelly(2 requests at once):
```shell
$ time seq 10  | xargs -n 1 -P 2 bash -c 'url="http://127.0.0.1:8080/predictions/bert"; curl -X POST $url -T input.txt'

0.05s user 0.08s system 0% cpu 19.611 total
```
That took half the time! 

Let's push it further with 4 requests at once, to exploit queue benefits:

Parallelly(4 requests at once):
```shell
$ time seq 10  | xargs -n 1 -P 4 bash -c 'url="http://127.0.0.1:8080/predictions/bert"; curl -X POST $url -T input.txt'

0.04s user 0.08s system 0% cpu 19.465 total
```
That did not speed up, but we get similar performance!

Let's now scale up to 4 workers.

#### 4 workers
```shell
$ curl -v -X PUT "http://localhost:8081/models/bert?min_worker=2&max_worker=2&synchronous=true"

*   Trying 127.0.0.1:8081...
* Connected to localhost (127.0.0.1) port 8081 (#0)
> PUT /models/bert?min_worker=4&max_worker=4&synchronous=true HTTP/1.1
> Host: localhost:8081
> User-Agent: curl/7.87.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: 7f7d7eb5-4488-4294-bfc1-8a6f03daa81a
< Pragma: no-cache
< Cache-Control: no-cache; no-store, must-revalidate, private
< Expires: Thu, 01 Jan 1970 00:00:00 UTC
< content-length: 54
< connection: keep-alive
< 
{
  "status": "Workers scaled to 4 for model: bert"
}
* Connection #0 to host localhost left intact
```

4 Parallel requests:
```shell
$ time  seq 10  | xargs -n 1 -P 4 bash -c 'url="http://127.0.0.1:8080/predictions/bert"; curl -X POST $url -T input.txt'

0.04s user 0.08s system 0% cpu 12.030 total
```
A `1.5x` speedup from 2 workers, and a `3x` speedup from 1 worker.



## Conclusion

For 10 inferences:

|           | Sequential | Parallel (n_requests=n_jobs) |
|-----------|------------|------------------------------|
| 1 Worker  | 37.771s     | 38.201s                       |
| 2 Workers | 38.274s     | 19.611s                       |
| 4 Workers | 38.449s    | 12.030s                       |

Scaling up workers correcly can have upto `3.2x` speedups.