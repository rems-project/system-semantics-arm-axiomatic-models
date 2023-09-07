#!/bin/sh
./nightly_ifetch.sh \
    ../../../litmus-tests/litmus-tests-armv8a-system/tests-isla/ifetch/HAND/@all \
    --models ../models/aarch64_ifetch.cat ../models/aarch64_ifetch.cat ../models/aarch64_ifetch.cat ../models/aarch64_ifetch.cat CU_check.cat \
    --variants None IDC DIC IDC,DIC None --allow-bad-names -v