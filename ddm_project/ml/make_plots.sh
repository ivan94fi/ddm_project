for i in {0..5};
do
    python grid_search.py $i --iforest --ocsvm --plot_metrics
    echo "----------------------------"
done
