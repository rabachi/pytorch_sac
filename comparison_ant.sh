for i in {1..4}
do
	sbatch vaughan_script.sh Ant-v3 1
done

# for i in {1..4}
# do
# 	sbatch sac_test.sh sunrise_sac Hopper-v3 True False
# done

for i in {1..4}
do
	sbatch vaughan_script.sh Ant-v3 5
done

for i in {1..4}
do
	sbatch vaughan_script.sh Ant-v3 10
done

# for i in {1..4}
# do
# 	sbatch sac_test.sh sac Hopper-v3 True False
# done
