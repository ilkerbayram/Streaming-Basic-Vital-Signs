#include <Streaming.h>
#include "FspTimer.h"

FspTimer audio_timer;
uint64_t count=0;
uint64_t start_time=0;

bool new_sample; // used to indicate if new samples are available
float sampling_rate = 200.0f; // sampling rate for the analog pins, in Hz
unsigned long baudrate = 115200u;


// definition of the ADC pins
int sensor1_pin = A0;
int sensor2_pin = A2;
int sensor3_pin = A4;

// to hold the samples
int signal1;
int signal2;
int signal3;

// callback method used by timer
void timer_callback(timer_callback_args_t __attribute((unused)) *p_args) {

  signal1 = analogRead(sensor1_pin);
  signal2 = analogRead(sensor2_pin);
  signal3 = analogRead(sensor3_pin);
  new_sample = 1;
}

bool beginTimer(float rate) {
  uint8_t timer_type = GPT_TIMER;
  int8_t tindex = FspTimer::get_available_timer(timer_type);
  if (tindex < 0){
    tindex = FspTimer::get_available_timer(timer_type, true);
  }
  if (tindex < 0){
    return false;
  }

  FspTimer::force_use_of_pwm_reserved_timer();

  if(!audio_timer.begin(TIMER_MODE_PERIODIC, timer_type, tindex, rate, 0.0f, timer_callback)){
    return false;
  }

  if (!audio_timer.setup_overflow_irq()){
    return false;
  }

  if (!audio_timer.open()){
    return false;
  }

  if (!audio_timer.start()){
    return false;
  }
  return true;
}
void setup() {
  Serial.begin(baudrate);
  beginTimer(sampling_rate);
}


void loop() {
  if (new_sample)
  {
  new_sample = 0;
  Serial << -1 <<"," << signal1 << "," << signal2 << "," << signal3 << endl;
  }
}