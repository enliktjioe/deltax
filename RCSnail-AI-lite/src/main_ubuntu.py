import os
import time
import glob
import logging
import traceback
import asyncio
import signal
import numpy as np
import zmq
from zmq.asyncio import Context

from commons.common_zmq import recv_array_with_json, initialize_subscriber, initialize_publisher
from commons.configuration_manager import ConfigurationManager
from commons.car_controls import CarControlUpdates

from src.learning.model_wrapper import ModelWrapper
from src.learning.training.generator import Generator
from utilities.transformer import Transformer
from src.utilities.recorder import Recorder

import skimage
import skimage.transform
from skimage import io

#DELTAX COMMENT:
# This code assumes images that the model was trained with were downscaled with PIL.Image.NEAREST to size HxW: 120x180
# that images were cropped to remove visual input that is above the track wall, leaving just the bottom 60 rows


async def main_dagger(context: Context):
    config_manager = ConfigurationManager()
    conf = config_manager.config

    recorder = None 

    data_queue = context.socket(zmq.SUB)
    controls_queue = context.socket(zmq.PUB)

    control_mode = conf.control_mode # DELTAX: make sure this is full_model in the config

    init_jalan = True
    counter_speed = 0

    try:
        model = ModelWrapper(conf, output_shape=2)


        await initialize_subscriber(data_queue, conf.data_queue_port)
        await initialize_publisher(controls_queue, conf.controls_queue_port)

        while True:
            frame, data = await recv_array_with_json(queue=data_queue)

            frame = frame  #/255.0  <- DELTAX make sure you trained the model on same range of data
            #print(np.max(frame))  <- if incoming frames are in range 0-255 and you trained the model on images scaled to 0-1, model wont work
            frame = frame[:,::-1,:]  #DELTAX: image comes in mirrored in Ubuntu! Need to flip them back for data to be like what we tained with 
            frame = np.flip(frame, axis=2)            
            expert_action = data 

            if np.random.random()>0.99:
                skimage.io.imsave("raw_input_in_try_loop.png",frame)


            if frame is None or expert_action is None:
                logging.info("None data")
                continue

            #DELTAX PREPROCESSING like the one that was done when training the model
            mem_frame = frame[-60:,:,:].reshape(1,60,180,3) #reshaping as model expects a minibatch dimension as first dim

            #DELTAX - spy function that saves images so you can peek what the model actually sees
            if np.random.random()>0.99:
                skimage.io.imsave("example_input_in_try_loop.png",mem_frame[0,:,:,:])

            
            if mem_frame is None:
                # Send back these first few instances, as the other application expects 1:1 responses
                print("NONE NONE NONE NONE")
                controls_queue.send_json({'d_steering':0,'d_gear':1,'d_throttle':0.65}) #TODO need to send repetition of last command or {'d_steer':0,....}
                continue


            try:
                if control_mode == 'full_expert': #this would be if you steer by controller
                    next_controls = expert_action.copy()
                    time.sleep(0.035)
                elif control_mode == 'full_model': #DELTAX: this is where we work in!
                    controls = model.model.predict(mem_frame)[0] #DELTAX - neural network makes predictions based on frame
                    #DELTAX: this printout helps you understand if model outputs are in reasonable range (-1 to 1 fr steering)

                    s = max(-1,min(1, np.float64(controls[0])))
                    #DELTAX: floats we sent must be float64, as flat32 is not json serializable for some reason
                    next_controls={'d_steering': s}
    
                    next_controls['d_gear'] = 1 # DELTAX: always go forward
                    #for throttle you can use 0 to just test if car turns wheels in good direction at different locations
                    #the minimal throttle to make the car move slowly is around 0.65, depends on battery charge level
                    
                    t = 0.49

                    if (s >= -0.25) and (s<=0.25):
                        counter_speed = counter_speed + 1
                    else:
                        counter_speed = 0
                        #pass

                    if counter_speed >=5:
                        t = 0.8
                        counter_speed = 0

                    if init_jalan:
                        next_controls['d_throttle'] = np.float64(t) # max(0,min(1, np.float64(controls[1])))}
                        init_jalan=False
                    else:
                        next_controls['d_throttle'] = np.float64(t) # max(0,min(1, np.float64(controls[1])))}
                    
                    print(next_controls, counter_speed)


                    #DELTAX: to use model's output for throttle, not fixed value
                    #next_controls['d_throttle'] = max(0,min(1, np.float64(controls[1])))}
                
                else:
                    raise ValueError('Misconfigured control mode!')

                controls_queue.send_json(next_controls)

            except Exception as ex:
                print("Predicting exception: {}".format(ex))
                traceback.print_tb(ex.__traceback__)
    except Exception as ex:
        print("Exception: {}".format(ex))
        traceback.print_tb(ex.__traceback__)
    finally:
        data_queue.close()
        controls_queue.close()

        files = glob.glob(conf.path_to_session_files + '*')
        for f in files:
            os.remove(f)
        logging.info("Session partials deleted successfully.")

        if recorder is not None:
            #recorder.save_session_with_expert()
            recorder.save_session_with_predictions()

        model.save_best_model()


def signal_cancel_tasks(loop):
    loop = asyncio.get_event_loop()
    for task in asyncio.Task.all_tasks(loop):
        task.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    loop = asyncio.get_event_loop()
    #loop.add_signal_handler(signal.SIGINT, cancel_tasks, loop)
    #loop.add_signal_handler(signal.SIGTERM, cancel_tasks, loop)
    signal.signal(signal.SIGINT, signal_cancel_tasks)
    signal.signal(signal.SIGTERM, signal_cancel_tasks)
    context = zmq.asyncio.Context()
    try:
        loop.run_until_complete(main_dagger(context))
    except Exception as ex:
        logging.error("Base interruption: {}".format(ex))
        traceback.print_tb(ex.__traceback__)
    finally:
        loop.close()
        context.destroy()
