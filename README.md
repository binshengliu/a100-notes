# A100@RMIT

1. Only `PyTorch>=1.7.1 and CUDAToolkit>=11.0` works. Always install [the latest
   version](https://pytorch.org/get-started/locally/) if possible. Check out the
   [documentation](https://pytorch.org/get-started/previous-versions/) if
   non-latest versions are needed.

   An example of installing PyTorch==1.7.1:
   ```
   conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
   ```

2. Limit PyTorch internal threads. This server has 256 processing units with
   hyperthreading. PyTorch would start 256 threads per process by default which
   degrades overall performance due to the overhead. Even worse it might
   **crash** the server and this has happened before.

   Put this line in the very beginning of your script or entry function:
   ```
   torch.set_num_threads(min(torch.get_num_threads(), 16))
   ```

3. Similarly, explicitly set the number of processes when using parallelization
   like
   [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html)
   and
   [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html).
   The parameter name might be `cpus`, `processes`, `threads`, `workers`, etc.

   ```
   import multiprocessing as mp


   def n_process() -> int:
       '''Use half of CPUs but no more than 20'''
       return min(max(mp.cpu_count() // 2, 1), 20)


   def train() -> None:
       with mp.Pool(processes=n_process()) as pool:
           # do stuff
           pool.imap(...)
   ```
