#!/usr/bin/python
from __future__ import with_statement
import os
import sys
import time
import random
import logging
#import urllib3
import argparse
import threading
import numpy as np

class WebStats( object ):
    def __init__( self, tot_bytes=0, tot_requests=0, tot_errors=0,
                  avg_lat=0.0, duration=0.0, bw=0.0, stddev_lat=0.0,
                  min_lat=0.0, max_lat=0.0 ):
        self.tot_bytes    = tot_bytes
        self.tot_requests = tot_requests
        self.tot_errors   = tot_errors
        self.avg_lat      = avg_lat
        self.stddev_lat   = stddev_lat
        self.min_lat      = min_lat
        self.max_lat      = max_lat
        self.duration     = duration
        self.bw           = bw
        
        return

class ThreadStats( object ):
    def __init__( self, requests=[], responses=[], byte_count=[], errors=[], 
                  avg_lat=[], duration=0.0 ):
        self.requests   = requests
        self.responses  = responses
        self.byte_count = byte_count
        self.errors     = errors
        self.avg_lat    = avg_lat
        self.duration   = duration
        return

class TestParam( object ):
    def __init__( self, host="", port=0, threads=1, http_pool=None, base_url="", ramp=1,
                  duration=1, conns=1, rand_req=False, max_rand_obj=1, req_dist="", poisson_lam=1.0,
                  gauss_mean=1.0, gauss_std=1.0, max_iters=131072 ):
        self.host         = host
        self.port         = port
        self.threads      = threads
        self.http_pool    = http_pool
        self.base_url     = base_url
        self.ramp         = ramp
        self.duration     = duration
        self.conns        = conns
        self.rand_req     = rand_req
        self.max_rand_obj = max_rand_obj
        self.req_dist     = req_dist
        self.poisson_lam  = poisson_lam
        self.gauss_mean   = gauss_mean
        self.gauss_std    = gauss_std
        self.max_iters    = max_iters
        return

def calc_web_stats( thread_stats ):
    """ Calculate:
        Average latency per thread
        Minimum latency over all threads
        Maximum latency over all threads
        Standard deviation in latency
        Total transfer size
        Bandwidth
        Total number of requests
        Number of requests per second
        Total number of errors
    """
    # Reorganize thread statistics for processing
    thread_stats.requests   = np.array( thread_stats.requests   )
    thread_stats.responses  = np.array( thread_stats.responses  )
    thread_stats.byte_count = np.array( thread_stats.byte_count )
    thread_stats.errors     = np.array( thread_stats.errors     )
    thread_stats.avg_lat    = np.array( thread_stats.avg_lat    )
    
    # Calculate average latency per thread
    avg_lat    = np.average( thread_stats.avg_lat )
    # Calculate standard deviation in latency
    stddev_lat = np.std( thread_stats.avg_lat     ) * 100.0
    # Calculate minimum latency over all threads
    min_lat    = np.amin( thread_stats.avg_lat )
    # Calculate maximum latency over all threads
    max_lat    = np.amax( thread_stats.avg_lat )
    # Calculate total transfer size
    tot_bytes  = np.sum( thread_stats.byte_count  )
    # Caclulate bandwidth
    bw           = np.divide( tot_bytes, thread_stats.duration    )
    # Calculate total number of requests
    tot_requests = np.sum( thread_stats.requests  )
    # Calculate number of requests per second
    req_p_s      = np.divide( tot_requests, thread_stats.duration )
    # Calculate total number of errors
    tot_errors   = np.sum( thread_stats.errors )

    web_stats = WebStats( tot_bytes=tot_bytes, tot_requests=tot_requests, tot_errors=tot_errors,
                          avg_lat=avg_lat, stddev_lat=stddev_lat, min_lat=min_lat, max_lat=max_lat,
                          duration=thread_stats.duration, bw=bw )
    
    return web_stats

def convert_units( web_stats ):
    """ Adjust metrics to appropriate units """
    ms_p_s = 1000.0
    MB_p_B = 1024 * 1024
    bits_p_B = 8

    web_stats.tot_bytes /= MB_p_B

    web_stats.avg_lat *= ms_p_s
    web_stats.min_lat *= ms_p_s
    web_stats.max_lat *= ms_p_s

    web_stats.bw = (web_stats.bw / MB_p_B) * bits_p_B

    return web_stats

def write_csv( csv_file, web_stats, main_args ):
    """ Generate CSV file """
    hdr_fmt = "%s - %s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
    header = hdr_fmt %  ( "Threads",
                          "Connections",
                          "Total Transfer Size (MB)",
                          "Total Requests",
                          "Total Errors",
                          "Avg. Latency (ms)",
                          "Std. Dev. Latency (%)",
                          "Min. Latency (ms)",
                          "Max. Latency (ms)",
                          "Total Duration (s)",
                          "Bandwidth (Mbps)",
    )    
    mode = "w"
    if os.path.isfile( csv_file ):
        mode = "a"
    body_fmt = "%s - %s,%.2f,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"
    next_line = body_fmt % ( main_args.threads,
                             main_args.conns,
                             web_stats.tot_bytes,
                             web_stats.tot_requests,
                             web_stats.tot_errors,
                             web_stats.avg_lat,
                             web_stats.stddev_lat,
                             web_stats.min_lat,
                             web_stats.max_lat,
                             web_stats.duration,
                             web_stats.bw,
                           )

    with open( csv_file, mode ) as output_file:
        if mode == "w":
            output_file.write( header )
        output_file.write( next_line )
        
    return 
                        
def size_based_test( test_param, thread_stats, start_flag, stop_flag ):
    """ Size-based test to be carried out by each thread """

    name = threading.currentThread().getName()
    j = int( name )
    prefix_url = "%s/t_%s" % ( test_param.base_url, name )
    np.random.seed( j )
    http_pool = test_param.http_pool

    # Pre-stage requests and wait times
    sleep_times = [test_param.gauss_mean]    
    if test_param.req_dist == "gauss":
        sleep_times = np.abs( np.random.normal(loc=test_param.gauss_mean,
                                               scale=test_param.gauss_std, size=test_param.max_iters) )
    elif test_param.req_dist == "poisson":
        sleep_times = np.random.poisson( lam=test_param.poisson_lam, size=test_param.max_iters )
    urls = []
    if test_param.rand_req:
        for i in range( test_param.max_iters ):
            urls.append( "%s_%010d.html" % (prefix_url, np.random.randint(1, test_param.max_rand_obj)) )
    else:
        for i in range( test_param.max_iters ):
            urls.append( "%s_%010d.html" % (prefix_url, i + 1) )

    # Wait for start signal
    with start_flag:
        start_flag.wait()    
        logging.debug( "Starting" )

    i = 0
    while not stop_flag.isSet():
        # Wait before making next request
        #time.sleep( sleep_times[i] )
        req_start = time.time()
        try:
            rsp = http_pool.request( "GET", urls[i] )
            thread_stats.avg_lat[j] += time.time() - req_start
            thread_stats.responses[j] += 1
            thread_stats.byte_count[j] += len( rsp.data )
        except Exception, e:
            logging.debug( "Error while requesting: %s - %s" % (urls[i], str(e)) )
            thread_stats.errors[j] += 1
        i += 1
    thread_stats.requests[j] = http_pool.num_requests
    if thread_stats.requests[j] > 0:
        thread_stats.avg_lat[j] = thread_stats.avg_lat[j] / float( thread_stats.requests[j] )
    logging.debug( "Exiting" )

    return

def duration_based_test( test_param, thread_stats, start_flag, stop_flag ):
    """ Duration-based test to be carried out by each thread """

    name = threading.currentThread().getName()
    j = int( name )
    prefix_url = "%s/t_%s" % ( test_param.base_url, name )
    np.random.seed( j )
    # http_pool = test_param.http_pool

    # Pre-stage requests and wait times
    sleep_times = [test_param.gauss_mean]
    if test_param.req_dist == "gauss":
        sleep_times = np.abs( np.random.normal(loc=test_param.gauss_mean,
                                               scale=test_param.gauss_std, size=test_param.max_iters) )
    elif test_param.req_dist == "poisson":
        sleep_times = np.random.poisson( lam=test_param.poisson_lam, size=test_param.max_iters )
    urls = []
    if test_param.rand_req:
        for i in range( test_param.max_iters ):
            urls.append( "%s_%010d.html" % (prefix_url, np.random.randint(1, test_param.max_rand_obj)) )
    else:
        for i in range( test_param.max_iters ):
            urls.append( "%s_%010d.html" % (prefix_url, i + 1) )

    # Wait for start signal
    logging.debug( "Waiting for start event" )
    event_start = start_flag.wait()
    logging.debug( "Event %s: Starting" , event_start )
    start = time.time()
    
    i = 0 
    while not stop_flag.isSet():
        dt = time.time() - start
        # Wait before making next request
        # time.sleep( sleep_times[i] )
        req_start = time.time()            
        try:
            # rsp = http_pool.request( "GET", urls[i] )
            if dt > test_param.ramp:          
                thread_stats.avg_lat[j] += time.time() - req_start            
                thread_stats.responses[j] += 1
                thread_stats.byte_count[j] += len( rsp.data )
        except Exception, e:
            logging.debug( "Error while requesting: %s - %s" % (urls[i], str(e)) )
            if dt > test_param.ramp:
                thread_stats.errors[j] += 1
        i += 1
    # thread_stats.requests[j] = http_pool.num_requests
    if thread_stats.requests[j] > 0:
        thread_stats.avg_lat[j] = thread_stats.avg_lat[j] / float( thread_stats.requests[j] )        
    logging.debug( "Exiting" )

    return

def main( ):
    """ Main function """

    # Parse command line arguments
    parser = argparse.ArgumentParser( )
    parser.add_argument( "--threads", dest="threads", type=int, default=1,
                         help="Number of threads to use"  )
    parser.add_argument( "--host", dest="host", default="localhost",
                         help="Web server host name" )
    parser.add_argument( "--port", dest="port", type=int, default=8080,
                         help="Web server port number" )
    parser.add_argument( "--duration", dest="duration", type=float, default=60.0,
                         help="Duration of test in seconds" )
    parser.add_argument( "--ramp", dest="ramp", type=float, default=30.0,
                         help="Ramp time for duration-based testing" )
    parser.add_argument( "--size", dest="transfer_size", type=int, default=1024,
                         help="Total transfer size in bytes. Overrides duration-based settings" )
    parser.add_argument( "--connections", dest="conns", type=int, default=1,
                         help="Number of connections to use per thread" )
    parser.add_argument( "--test-type", dest="test_type", choices=["duration","size"], default="duration",
                         help="Type of test to perform" )
    parser.add_argument( "--random", dest="rand_req", action="store_true", default=False,
                         help="Indicates that threads should perform random requests. Otherwise sequential requests are performed." )
    parser.add_argument( "--max-rand-obj", dest="max_rand_obj", type=int, default=1000,
                         help="Maximum number of objects from which clients will make random requests" )
    parser.add_argument( "--output-dir", dest="output_dir", default=os.path.dirname(os.path.realpath(__file__)),
                         help="Directory to store output CSV file." )
    parser.add_argument( "--req-dist", dest="req_dist", choices=["gauss", "poisson"], default="gauss",
                         help="Client wait time distribution type." )
    main_args = parser.parse_args( )

    # Setup logging
    logging.basicConfig( level=logging.DEBUG,
                         format="[%(threadName)s]: [%(asctime)-15s] - %(message)s",
                         filename="traffic_gen.log" )
    logging.getLogger( "urllib3" ).setLevel( logging.WARNING )

    # Setup thread arguments
    base_url = "http://%s:%s/test" % ( main_args.host, main_args.port )
    csv_file = os.path.join( main_args.output_dir, "http_benchmark.csv" )
    start_flag = threading.Event()
    stop_flag = threading.Event()
    gauss_mean = 1.0 / 16384.0
    gauss_std = 0.5
    poisson_lam = gauss_mean
    
    thread_stats = ThreadStats()
    for i in range( main_args.threads ):
        thread_stats.requests.append( 0 )
        thread_stats.responses.append( 0 )
        thread_stats.byte_count.append( 0 )
        thread_stats.errors.append( 0 )
        thread_stats.avg_lat.append( 0.0 )
    
    #http_pool = urllib3.connectionpool.HTTPConnectionPool( main_args.host,
    #                                                       port=main_args.port,
    #                                                       maxsize=(main_args.threads * main_args.conns) )
    if main_args.test_type == "size":
        target = size_based_test
        test_param = TestParam( http_pool=http_pool, host=main_args.host, port=main_args.port, threads=main_args.threads,
                                base_url=base_url, conns=main_args.conns, rand_req=main_args.rand_req, 
                                max_rand_obj=main_args.max_rand_obj, req_dist=main_args.req_dist,
                                gauss_mean=gauss_mean, gauss_std=gauss_std, poisson_lam=poisson_lam )
    else:
        target = duration_based_test
        test_param = TestParam( http_pool=http_pool, host=main_args.host, port=main_args.port, threads=main_args.threads,
                                base_url=base_url, ramp=main_args.ramp,
                                duration=main_args.duration, conns=main_args.conns, rand_req=main_args.rand_req,
                                max_rand_obj=main_args.max_rand_obj, req_dist=main_args.req_dist,
                                gauss_mean=gauss_mean, gauss_std=gauss_std, poisson_lam=poisson_lam )
        
    thread_args = ( test_param, thread_stats, start_flag, stop_flag )
                         
    # Spawn threads
    for i in range( main_args.threads ):
        next_name = "%03d" % ( i )
        next_thread = threading.Thread( name=next_name,
                                        target=target,
                                        args=thread_args,
                                      )
        next_thread.start()

    init_interval = 3
    logging.debug( "Waiting %d s for %d threads to initialize" % (init_interval, main_args.threads) )
    time.sleep( init_interval )
    logging.debug( "Signaling threads to start test" )
    start_flag.set()

    # Add ramp time

    # Wait for test completion
    start = time.time()
    if main_args.test_type == "duration":        
        sleep_time = main_args.ramp + main_args.duration
        logging.debug( "Waiting %d s for test to complete" % (sleep_time) )
        time.sleep ( sleep_time )
    else:
        logging.debug( "Waiting for %d B to be requested for test to complete" % (main_args.transfer_size) )
        sleep_time = gauss_mean * 10.0
        if main_args.req_dist == "poisson":
            sleep_time = poisson_lam * 10.0
        while np.sum(thread_stats.byte_count) < main_args.transfer_size:
            time.sleep( sleep_time )
    stop_flag.set()
            
    thread_stats.duration = time.time() - start
    logging.debug( "Test completed" )
    
    # Join on threads
    main_thread = threading.currentThread()
    for next_thread in threading.enumerate():
        if next_thread is not main_thread:
            next_thread.join( )

    # Calculate statistics
    web_stats = calc_web_stats( thread_stats )
    web_stats = convert_units( web_stats )

    # Save statistics to CSV file
    write_csv( csv_file, web_stats, main_args )
    logging.debug( "Wrote %s" % (csv_file) )
    
    return 0

if __name__ == "__main__":
    sys.exit( main() )
