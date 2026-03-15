#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

SparkFun_VL53L5CX myImager;
VL53L5CX_ResultsData measurementData;

const int IMAGE_WIDTH = 8;
const float MIN_SIGNAL_THRESHOLD = 0.1000; 

void setup() {
  // 1. Maximize Serial Throughput (250k or 500k is often more stable than 115k)
  Serial.begin(250000); 
  delay(500);
  
  // 2. Maximize I2C Bus Speed (Fast Mode Plus)
  Wire.begin();
  Wire.setClock(1000000); // 1MHz I2C (Check if your MCU/Pull-ups support this)

  if (myImager.begin() == false) {
    while (1);
  }

  myImager.setResolution(8 * 8); 

  // 3. Optimization: Minimum Integration Time
  // Reducing this increases frequency but lowers max range. 
  // 10ms allows for ~60Hz theoretical peak, though 15-20ms is safer for SNR.
  myImager.setIntegrationTime(35); 
  
  // 4. Set Ranging Frequency (Hz)
  // For 8x8, the hardware limit is usually 15Hz. Let's push for 15.
  myImager.setRangingFrequency(15);

  myImager.setSharpenerPercent(15); // Disable sharpening to save processing time
  myImager.startRanging();
}

void loop() {
  // Use the interrupt-driven data check
  if (myImager.isDataReady()) {
    if (myImager.getRangingData(&measurementData)) {
      
      // Removed heavy F() macros and headers for raw speed
      for (int i = 0; i < 64; i++) {
        // Direct access: No coordinate transforms unless necessary
        int d = measurementData.distance_mm[i];
        int s = measurementData.target_status[i];
        float flux = measurementData.signal_per_spad[i];

        // Fast ternary logic for filtering
        bool valid = (s == 5 && flux > MIN_SIGNAL_THRESHOLD);
        
        // Print raw values separated by commas for faster parsing
        Serial.print(valid ? d : 0);
        Serial.print((i % 8 == 7) ? "\n" : ",");
      }
      Serial.print(" \n"); // 'f' marker for Frame End
    }
  }
}