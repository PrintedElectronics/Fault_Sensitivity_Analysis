`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=2;
    parameter NUM_A=4;
    parameter WIDTH_A=4;
    parameter NUM_W=21;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=12;
    parameter NUM_B1=3;
    parameter WIDTH_B1=16;

localparam period = 250000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [75, -42, -55, -75, -13, -69, -105, 30, 71, 31, 108, -1, -15, 18, 5, -7, 0, -57, -62, 48, -29]
assign weights = {8'sb01001011, 8'sb11010110, 8'sb11001001, 8'sb10110101, 8'sb11110011, 8'sb10111011, 8'sb10010111, 8'sb00011110, 8'sb01000111, 8'sb00011111, 8'sb01101100, 8'sb11111111, 8'sb11110001, 8'sb00010010, 8'sb00000101, 8'sb11111001, 8'sb00000000, 8'sb11000111, 8'sb11000010, 8'sb00110000, 8'sb11100011};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [-17315, 17487, -11255, -1112, -698, 639]
assign biases = {16'sb1011110001011101, 16'sb0100010001001111, 16'sb1101010000001001, 12'sb101110101000, 12'sb110101000110, 12'sb001001111111};
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
