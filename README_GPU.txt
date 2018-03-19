1. In order to run ProjE on the Helix server first: 
$ ssh helix.jax.org

and then eneter your password.

2. log in to a gpu node to run an interactive job on GPU:

$ qsub -I -q gpu -l nodes=1:ppn=1:gpus=1

Note: You can just enter qsub -I -q gpu, but JAX IT asked us to specify number of nodes, processors and gpus.

Option: You can add -m abe to the above command by which you receive an email when the job is finished or killed. 


3. Go to the directory for Project 
$ cd/projects/robinson-lab/ProjE_JAX/ProjE-master


Note: If not unzipped, first unzip the ProjE-master.zip and data.zip. Move the unzipped data folder to the directory of the ProjE-master.


4. Activate tensorflow using:

$ source active tensorflow

Note: (tensorflow)$  # Your prompt should change 


5. Run the code using 

$ python ./ProjE_softmax.py --dim 200 --batch 200 --data ./data/FB15k/ --eval_per 1 --worker 3 --eval_batch 500 --max_iter 100 --generator 10

 
