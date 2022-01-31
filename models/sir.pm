ctmc

const double ki = 0.001;
const double kr = 0.1;

const int maxPop;
const int initS = maxPop-5;
const int initI = 5;
const int initR = 0;

module SIR

popS: [0..maxPop] init initS;
popI: [0..maxPop] init initI;
popR: [0..maxPop] init initR;

[]popS > 0 & popI > 0 & popI < maxPop  ->    ki*popS*popI  : (popS'= popS-1) & (popI'= popI+1);
[]popI > 0 & popR < maxPop ->    kr*popI  : (popR'= popR+1) & (popI'= popI-1);
[]popI=0 -> 1 : true;

endmodule

rewards
  (popI > 50) : 1;
endrewards