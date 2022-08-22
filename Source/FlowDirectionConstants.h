#ifndef FLOW_DIRECTION_CONSTANTS_H
#define FLOW_DIRECTION_CONSTANTS_H


const unsigned char DIRECTION_NONE = 0;
const unsigned char DIRECTION_RIGHT = 1;
const unsigned char DIRECTION_DOWN_RIGHT = 2;
const unsigned char DIRECTION_DOWN = 4;
const unsigned char DIRECTION_DOWN_LEFT = 8;
const unsigned char DIRECTION_LEFT = 16;
const unsigned char DIRECTION_UP_LEFT = 32;
const unsigned char DIRECTION_UP = 64;
const unsigned char DIRECTION_UP_RIGHT = 128;

const unsigned char DIRECTION_CODE[3][3] = {{DIRECTION_UP_LEFT, DIRECTION_UP, DIRECTION_UP_RIGHT},
                                            {DIRECTION_LEFT, DIRECTION_NONE, DIRECTION_RIGHT},
                                            {DIRECTION_DOWN_LEFT, DIRECTION_DOWN, DIRECTION_DOWN_RIGHT}};


#endif
