`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=2;
    parameter NUM_A=6;
    parameter WIDTH_A=4;
    parameter NUM_W=27;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=10;
    parameter NUM_B1=3;
    parameter WIDTH_B1=11;

localparam period = 320000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [51, 1, -2, -50, 3, -7, -31, 0, 7, 88, -15, -3, -1, -1, -3, -3, -2, 1, 2, -2, -2, 38, -30, -17, -8, 22, -13]
assign weights = {8'sb00110011, 8'sb00000001, 8'sb11111110, 8'sb11001110, 8'sb00000011, 8'sb11111001, 8'sb11100001, 8'sb00000000, 8'sb00000111, 8'sb01011000, 8'sb11110001, 8'sb11111101, 8'sb11111111, 8'sb11111111, 8'sb11111101, 8'sb11111101, 8'sb11111110, 8'sb00000001, 8'sb00000010, 8'sb11111110, 8'sb11111110, 8'sb00100110, 8'sb11100010, 8'sb11101111, 8'sb11111000, 8'sb00010110, 8'sb11110011};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [-759, 750, -903, -71, -51, 331]
assign biases = {11'sb10100001001, 11'sb01011101110, 11'sb10001111001, 10'sb1110111001, 10'sb1111001101, 10'sb0101001011};
top DUT(.inp(inp),
        .weights(weights),
        .biases(biases),
        .out(out)
        );


    integer inFile,outFile,i;
    initial
    begin
        $display($time, " << Starting the Simulation >>");
            inFile = $fopen("./sim/sim.Xtest","r");
        if (inFile == `NULL) begin
                $display($time, " file not found");
                $finish;
        end
        outFile = $fopen("./sim/output.txt");
        while (!$feof(inFile)) begin
            for (i=0;i<NUM_A;i=i+1) begin
                $fscanf(inFile,"%d ",at[i]);
            end
            $fscanf(inFile,"\n");
            #(period)
            $fwrite(outFile,"%d\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end


    genvar gi;
    generate
    for (gi=0;gi<NUM_A;gi=gi+1) begin : genbit
        assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
    end
    endgenerate


    endmodule
