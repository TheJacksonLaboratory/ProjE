1. In order to run ProjE on the Helix server first: 
$ ssh helix.jax.org

and then eneter your password.

2. log in to a gpu node to run an interactive job on GPU:

$ qsub -I -q gpu -l nodes=1:ppn=13
Note: We are still investigating the optimal number of processors. So far, we have not noticed
significant improvement in the running time by increasing the number of processors.


3. Go to the directory of ProjE
$ cd/projects/robinson-lab/ProjE_JAX/ProjE-master


Note: If not unzipped, first unzip the ProjE-master.zip and data.zip. Move the unzipped data folder to the directory of the ProjE-master.


4. Activate tensorflow using:

$ source activate tensorflow

Note: (tensorflow)$  # Your prompt should change 


5. Run the code using 

$ python ./ProjE_softmax.py --dim 200 --batch 200 --data ./data/FB15k/ --eval_per 1 --worker 3 --eval_batch 500 --max_iter 100 --generator 10

Note: you may change the dataset and choose ./data/WN18.

 
