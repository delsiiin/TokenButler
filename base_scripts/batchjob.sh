#!/bin/bash

INPUT_FILE="alleval.sh"

while IFS= read -r CMD; do
    cat <<EOF > temp_job.slurm
#!/bin/bash
#SBATCH -J batcheval
#SBATCH --output=/home/ya255/projects/TokenButler/slurmlogs/%x_%j.out
#SBATCH --error=/home/ya255/projects/TokenButler/slurmlogs/%x_%j.err
#SBATCH --account=abdelfattah
#SBATCH --gpus=1
#SBATCH --mem=120000
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=16:00:00  # Adjust as needed

# Load any required modules

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate TokenButler

cd /home/ya255/projects/TokenButler/

echo "Running: $CMD"
$CMD
EOF

    sbatch temp_job.slurm
    rm temp_job.slurm  # Clean up temporary script
done < "$INPUT_FILE"
