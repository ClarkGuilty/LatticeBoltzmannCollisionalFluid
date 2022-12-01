#!/bin/bash
mkdir -p slurms
rm slurms/*
for nprocs in $(eval echo {$1..$2..$3})
do
echo $nprocs
echo "#!/bin/bash" > slurms/execute_$nprocs
echo "#SBATCH -n $nprocs" >> slurms/execute_$nprocs
echo "#SBATCH -A phys-743" >> slurms/execute_$nprocs
echo "#SBATCH --reservation phys-743" >> slurms/execute_$nprocs
echo "#SBATCH -o out_$nprocs.out" >> slurms/execute_$nprocs
echo "export PATH=/work/scitas-share/julia-1.8.3/bin:\$PATH" >> slurms/execute_$nprocs
echo "export LD_LIBRARY_PATH=/work/scitas-share/julia-1.8.3/lib:/work/scitas-share/julia-1.8.3/lib64:\$LD_LIBRARY_PATH" >> slurms/execute_$nprocs
echo "module load curl" >> slurms/execute_$nprocs
echo "srun -n $nprocs julia --project=. buildingSimulation.jl" >> slurms/execute_$nprocs
done
