//generic design, we don't actually know the weights and intercepts, they are given to the network through the testbench.
//weights: [[[88, 86, -88, -86], [59, 57, -59, -59], [-12, -3, -6, -12]], [[-98, 72, 12], [1, 55, -4], [33, -72, 11]]]
//intercepts: [[-1, 571, -164], [-38551, -33633, 33375]]
module top (inp, weights, biases, out);
input [15:0] inp;
input [167:0] weights;
input [83:0] biases;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight 88 : 8'b01011000
    wire signed [12-1:0] n_0_0_po_0;
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[8-1:0]);

    //weight 86 : 8'b01010110
    wire signed [12-1:0] n_0_0_po_1;
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[16-1:8]);

    //weight abs(-88) : 8'b01011000
    wire signed [12-1:0] n_0_0_po_2;
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[24-1:16]);

    //weight abs(-86) : 8'b01010110
    wire signed [12-1:0] n_0_0_po_3;
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[32-1:24]);

    //accumulate products
    wire signed [15:0] n_0_0_sum;
    assign n_0_0_sum = $signed(biases[11-1:0]) + $signed(n_0_0_po_0) + $signed(n_0_0_po_1) + $signed(n_0_0_po_2) + $signed(n_0_0_po_3);
    //relu
    wire [14:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight 59 : 8'b00111011
    wire signed [12-1:0] n_0_1_po_0;
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[40-1:32]);

    //weight 57 : 8'b00111001
    wire signed [12-1:0] n_0_1_po_1;
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[48-1:40]);

    //weight abs(-59) : 8'b00111011
    wire signed [12-1:0] n_0_1_po_2;
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[56-1:48]);

    //weight abs(-59) : 8'b00111011
    wire signed [12-1:0] n_0_1_po_3;
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[64-1:56]);

    //accumulate products
    wire signed [15:0] n_0_1_sum;
    assign n_0_1_sum = $signed(biases[22-1:11]) + $signed(n_0_1_po_0) + $signed(n_0_1_po_1) + $signed(n_0_1_po_2) + $signed(n_0_1_po_3);
    //relu
    wire [14:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    //weight abs(-12) : 8'b00001100
    wire signed [12-1:0] n_0_2_po_0;
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[72-1:64]);

    //weight abs(-3) : 8'b00000011
    wire signed [12-1:0] n_0_2_po_1;
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[80-1:72]);

    //weight abs(-6) : 8'b00000110
    wire signed [12-1:0] n_0_2_po_2;
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[88-1:80]);

    //weight abs(-12) : 8'b00001100
    wire signed [12-1:0] n_0_2_po_3;
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[96-1:88]);

    //accumulate products
    wire signed [15:0] n_0_2_sum;
    assign n_0_2_sum = $signed(biases[33-1:22]) + $signed(n_0_2_po_0) + $signed(n_0_2_po_1) + $signed(n_0_2_po_2) + $signed(n_0_2_po_3);
    //relu
    wire [14:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    //weight abs(-98) : 8'b01100010
    wire signed [21-1:0] n_1_0_po_0;
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[104-1:96]);

    //weight 72 : 8'b01001000
    wire signed [21-1:0] n_1_0_po_1;
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[112-1:104]);

    //weight 12 : 8'b00001100
    wire signed [21-1:0] n_1_0_po_2;
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[120-1:112]);

    //accumulate products
    wire signed [23:0] n_1_0_sum;
    assign n_1_0_sum = $signed(biases[50-1:33]) + $signed(n_1_0_po_0) + $signed(n_1_0_po_1) + $signed(n_1_0_po_2);
    //relu
    wire [22:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    //weight 1 : 8'b00000001
    wire signed [21-1:0] n_1_1_po_0;
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[128-1:120]);

    //weight 55 : 8'b00110111
    wire signed [21-1:0] n_1_1_po_1;
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[136-1:128]);

    //weight abs(-4) : 8'b00000100
    wire signed [21-1:0] n_1_1_po_2;
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[144-1:136]);

    //accumulate products
    wire signed [23:0] n_1_1_sum;
    assign n_1_1_sum = $signed(biases[67-1:50]) + $signed(n_1_1_po_0) + $signed(n_1_1_po_1) + $signed(n_1_1_po_2);
    //relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight 33 : 8'b00100001
    wire signed [21-1:0] n_1_2_po_0;
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[152-1:144]);

    //weight abs(-72) : 8'b01001000
    wire signed [21-1:0] n_1_2_po_1;
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[160-1:152]);

    //weight 11 : 8'b00001011
    wire signed [21-1:0] n_1_2_po_2;
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[168-1:160]);

    //accumulate products
    wire signed [23:0] n_1_2_sum;
    assign n_1_2_sum = $signed(biases[84-1:67]) + $signed(n_1_2_po_0) + $signed(n_1_2_po_1) + $signed(n_1_2_po_2);
    //relu
    wire [22:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [22:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [22:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule
