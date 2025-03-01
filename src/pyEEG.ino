/*
EEG-8
Скетч оцифровывает сигналы c восьми сенсоров электроэнцефалограммы (ЭЭГ, EEG), устройства синхронизации, а также формирует тестовый сигнал "пила".  
Справочник по командам языка Arduino: http://arduino.ru/Reference 
*/

#include <TimerOne.h> // подключаем библиотеку TimerOne для задействования функций Таймера1 
/* предварительно данную библиотеку надо установить, для чего скачиваем ее 
на странице https://bitronicslab.com/neuromodelist, распаковываем архив и помещаем папку TimerOne
внутрь папки libraries, находящейся тут: "Мои документы/Arduino/libraries" 
Подробнее о TimerOne см. тут: http://robocraft.ru/blog/arduino/614.html */

byte i = 0;
int val;
String dat;
// функция sendData вызывается каждый раз, когда срабатывает прерывание Таймера1 (проходит заданное число микросекунд)
void sendData() {
  dat = "";
  // Оцифровка сигнала с аналогового входа А0. Отправка данных. 
                                            // всего в этой программе 4 поля, которые имеют имена A0, A1, A2, A3 (сверху вниз, по порядку их 
                                            // расположения в окне программы)
  val = analogRead(A0);                     // записываем в переменную val оцифрованное значение сигнала со входа А0 на Arduino.
                                            // val может принимать значение в диапазоне от 0 до 1023, см. http://arduino.ru/Reference/AnalogRead 
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,
                                            // см. описание команды map:  http://arduino.ru/Reference/Map 

  // Оцифровка сигнала с аналогового входа А1. Отправка данных. 
  val = analogRead(A1);                     
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,

  
  // Оцифровка сигнала с аналогового входа А2. Отправка данных.                                           
  val = analogRead(A2);                    
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,
 

  // Оцифровка сигнала с аналогового входа А3. Отправка данных. 
  val = analogRead(A3);                    
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,
                

  // Оцифровка сигнала с аналогового входа А4. Отправка данных. 
  val = analogRead(A4);                    
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,


  // Оцифровка сигнала с аналогового входа А5. Отправка данных. 
  val = analogRead(A5);                    
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,


  // Оцифровка сигнала с аналогового входа А6. Отправка данных. 
  val = analogRead(A6);                    
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,


  // Оцифровка сигнала с аналогового входа А7. Отправка данных. 
  val = analogRead(A7);                    
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,


  // Оцифровка сигнала с аналогового входа А8 (сигнал с устройства синхронизации). Отправка данных. 
  val = analogRead(A8);                    
  dat += map(val, 0, 1023, 0, 255).toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,


  // Формирование сигнала "пилы"
  dat += i.toString() + " ";  // записываем в Serial-порт значение val, предварительно отнормированное на диапазон значений от 0 до 255,
  Serial.println(dat);
  i++;                                             
}


// функция setup вызывается однократно при запуске Arduino
void setup() {
  Serial.begin(115200);                    // инициализируем Serial-порт на скорости 115200 Кбит/c. 
                                           // такую же скорость надо установить в программе для визуализации
  Timer1.initialize(3000);                 // инициализируем Таймер1, аргументом указываем интервал срабатывания - 3000 микросекунд 
                                           // (1 000 000 микросекунд = 1 сек)
  Timer1.attachInterrupt(sendData);        // как только проходит 3000 микросекунд - наступает прерывание (вызывается функция sendData)
}

void loop() {

}
