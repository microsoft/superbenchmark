#!/bin/bash

test_dir="$(dirname $0)"
test_cases=(
    test_deploy
    test_check_env
)

err_num=0
for test_case in ${test_cases[@]}; do
    ansible-playbook -i "${test_dir}/hosts.ini" "${test_dir}/tests/${test_case}.yaml"
    rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "tests ${test_case} PASSED"
    else
        echo "tests ${test_case} FAILED"
        err_num=$((err_num+1))
    fi
done

exit ${err_num}
