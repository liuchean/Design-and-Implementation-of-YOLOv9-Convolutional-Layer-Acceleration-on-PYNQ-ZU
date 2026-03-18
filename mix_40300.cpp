#include "mix_40300.h"




void mix_40300(
    const float input[bb][c1][input_height][input_width],
    const float weight[c2][c1][kernel_size_a][kernel_size_b],

	float output[bb][c2][output_height][output_width],

    float output1[bb][c2][output_height][output_width],
	float output2[bb][c2][output_height][output_width],
	float output3[bb][c2][output_height][output_width],
	float output4[bb][c2][output_height][output_width],
	int check1
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem

	#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem


    #pragma HLS INTERFACE m_axi port=output1 offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=output2 offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=output3 offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=output4 offset=slave bundle=gmem2



    #pragma HLS INTERFACE s_axilite port=input bundle=control
    #pragma HLS INTERFACE s_axilite port=weight bundle=control

	#pragma HLS INTERFACE s_axilite port=output bundle=control

    #pragma HLS INTERFACE s_axilite port=output1 bundle=control
	#pragma HLS INTERFACE s_axilite port=output2 bundle=control
	#pragma HLS INTERFACE s_axilite port=output3 bundle=control
	#pragma HLS INTERFACE s_axilite port=output4 bundle=control

	#pragma HLS INTERFACE s_axilite port=check1 bundle=control


	static fixed_t local_input_part1[bb][c1/2][input_height][input_width];
	static fixed_t local_input_part2[bb][c1/2][input_height][input_width];

	static fixed_ttt local_weight_part1[c2][c1/2][kernel_size_a][kernel_size_b];
	static fixed_ttt local_weight_part2[c2][c1/2][kernel_size_a][kernel_size_b];




    #pragma HLS RESOURCE variable=local_input_part1 core=RAM_2P_LUTRAM
	#pragma HLS RESOURCE variable=local_input_part2 core=RAM_2P_LUTRAM

//    #pragma HLS RESOURCE variable=local_weight_part1 core=RAM_2P_LUTRAM
//	#pragma HLS RESOURCE variable=local_weight_part2 core=RAM_2P_LUTRAM

	if(check1==0){
							for (int c = 0; c < c1; c++) {
								for (int i = 0; i < input_height; i++) {
									for (int j = 0; j < input_width; j++) {

										if (c < c1/2) {
											local_input_part1[0][c][i][j] = input[0][c][i][j];
										} else {
											local_input_part2[0][c - c1/2][i][j] = input[0][c][i][j];
										}
									}
								}
							}


							for (int co = 0; co < c2; co++) {
								for (int ci = 0; ci < c1; ci++) {
									for (int k = 0; k < kernel_size_a; k++) {
										for (int l = 0; l < kernel_size_b; l++) {

											if (ci < c1/2) {
												local_weight_part1[co][ci][k][l] = weight[co][ci][k][l];
											} else {
												local_weight_part2[co][ci - c1/2][k][l] = weight[co][ci][k][l];
											}
										}
									}
								}
							}

				}
		else if(check1==1){
			for (int c = 0; c < c1; c++) {
					for (int i = 0; i < input_height; i++) {
						for (int j = 0; j < input_width; j++) {

							if (c < c1/2) {
								local_input_part1[0][c][i][j] = input[0][c][i][j];
							} else {
								local_input_part2[0][c - c1/2][i][j] = input[0][c][i][j];
							}
						}
					}
				}


				for (int co = 0; co < c2; co++) {
					for (int ci = 0; ci < c1; ci++) {
						for (int k = 0; k < kernel_size_a; k++) {
							for (int l = 0; l < kernel_size_b; l++) {

								if (ci < c1/2) {
									local_weight_part1[co][ci][k][l] = weight[co][ci][k][l];
								} else {
									local_weight_part2[co][ci - c1/2][k][l] = weight[co][ci][k][l];
								}
							}
						}
					}
				}


		}
		else if(check1==2){
				for (int c = 0; c < 24; c++) {
										for (int i = 0; i < input_height; i++) {
											for (int j = 0; j < input_width; j++) {

												if (c < 24/2) {
													local_input_part1[0][c][i][j] = input[0][c][i][j];
												} else {
													local_input_part2[0][c - 24/2][i][j] = input[0][c][i][j];
												}
											}
										}
									}


									for (int co = 0; co < 24; co++) {
										for (int ci = 0; ci < 24; ci++) {
											for (int k = 0; k < kernel_size_a; k++) {
												for (int l = 0; l < kernel_size_b; l++) {

													if (ci < 24/2) {
														local_weight_part1[co][ci][k][l] = weight[co][ci][k][l];
													} else {
														local_weight_part2[co][ci - 24/2][k][l] = weight[co][ci][k][l];
													}
												}
											}
										}
									}

			}






	if(check1==1){
		conv2d_2448(local_input_part1, local_input_part2, local_weight_part1, local_weight_part2,
				 output1, output2,output3, output4);
	}
	else if(check1==2){

		conv2d_2424(local_input_part1, local_input_part2, local_weight_part1, local_weight_part2,
									 output1, output2,output3, output4);

		}


	else {
		conv2d_4848(local_input_part1, local_input_part2, local_weight_part1, local_weight_part2,
						 output1, output2,output3, output4);
	}

	for (int c_out = 0; c_out < c2; ++c_out) {
		for (int i = 0; i < output_height; ++i) {
			for (int j = 0; j < output_width; ++j) {
				output[0][c_out][i][j] = output1[0][c_out][i][j] + output2[0][c_out][i][j] + output3[0][c_out][i][j] + output4[0][c_out][i][j];
			}
		}
	}



}
