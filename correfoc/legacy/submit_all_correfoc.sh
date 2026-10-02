SUBMITTED=0
FAILED=0

for dir in test_pure_sc*/; do
    if [[ -f "$dir/job.sh" ]]; then
        job_id=$(cd "$dir" && sbatch job.sh 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "Submitted $dir -> $job_id"
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "FAILED $dir: $job_id"
            FAILED=$((FAILED + 1))
        fi
    fi
done

echo ""
echo "Done. Submitted: $SUBMITTED | Failed: $FAILED"
