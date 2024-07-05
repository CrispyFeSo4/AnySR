


echo '-------------set5-------------' &&
echo 'x2' &&
python test.py --config ./configs/test/test_anysr-set5-2.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x3' &&
python test.py --config ./configs/test/test_anysr-set5-3.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x4' &&
python test.py --config ./configs/test/test_anysr-set5-4.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&

echo '-------------set14-------------' &&
echo 'x2' &&
python test.py --config ./configs/test/test_anysr-set14-2.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x3' &&
python test.py --config ./configs/test/test_anysr-set14-3.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x4' &&
python test.py --config ./configs/test/test_anysr-set14-4.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&

echo '-------------b100-------------' &&
echo 'x2' &&
python test.py --config ./configs/test/test_anysr-b100-2.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x3' &&
python test.py --config ./configs/test/test_anysr-b100-3.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x4' &&
python test.py --config ./configs/test/test_anysr-b100-4.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&

echo '-------------urban100-------------' &&
echo 'x2' &&
python test.py --config ./configs/test/test_anysr-urban100-2.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x3' &&
python test.py --config ./configs/test/test_anysr-urban100-3.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x4' &&
python test.py --config ./configs/test/test_anysr-urban100-4.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&


echo '-------------manga109-------------' &&
echo 'x2' &&
python test.py --config ./configs/test/test_anysr-manga109-2.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x3' &&
python test.py --config ./configs/test/test_anysr-manga109-3.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&
echo 'x4' &&
python test.py --config ./configs/test/test_anysr-manga109-4.yaml --model $1 --mcell $2 --test_only $3 --entire_net $4&&


true
