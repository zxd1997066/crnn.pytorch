import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn

import argparse
import time
import os

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model_path', type=str, default='./data/crnn.pth', help='model_path')
parser.add_argument('--img_path', type=str, default='./data/demo.png', help='img_path')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='alphabet')
# for oob
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--precision', type=str, default='float32', help='precision')
parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--num_iter', type=int, default=1, help='num_iter')
parser.add_argument('--num_warmup', type=int, default=-1, help='num_warmup')
parser.add_argument('--profile', dest='profile', action='store_true', help='profile')
parser.add_argument('--quantized_engine', type=str, default=None, help='quantized_engine')
parser.add_argument('--ipex', dest='ipex', action='store_true', help='ipex')
parser.add_argument('--jit', dest='jit', action='store_true', help='jit')
args = parser.parse_args()
print ("args", args)


# evaluate
def evaluate(model, image, h2d_time):
    model.eval()
    total_time = 0.0
    total_sample = 0
    for i in range(args.num_iter):
        elapsed = time.time()
        preds = model(image)
        elapsed = time.time() - elapsed + h2d_time
        if args.profile:
            args.p.step()
        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
        if i >= args.num_warmup:
            total_time += elapsed
            total_sample += args.batch_size

    throughput = total_sample / total_time
    latency = total_time / total_sample * 1000
    print('inference latency: %.3f ms' % latency)
    print('inference Throughput: %f images/s' % throughput)

    return preds

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'crnn-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def main():
    # load model
    model = crnn.CRNN(32, 1, 37, 256)
    if args.device == 'cuda':
        model = model.cuda()
    print('loading pretrained model from %s' % args.model_path)
    model.load_state_dict(torch.load(args.model_path))

    converter = utils.strLabelConverter(args.alphabet)

    # load dataset
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(args.img_path).convert('L')
    image = transformer(image)
    h2d_time = time.time()
    if args.device == 'cuda':
        image = image.cuda()
    h2d_time = time.time() - h2d_time
    image = image.view(1, *image.size())
    image = Variable(image)
    # NHWC
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        image = image.contiguous(memory_format=torch.channels_last)
        print("---- Use NHWC model and input")

    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int(args.num_iter/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            args.p = p
            preds = evaluate(model, image, h2d_time)
    else:
        preds = evaluate(model, image, h2d_time)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

if __name__ == '__main__':

    if args.precision == "bfloat16":
        print('---- Enable AMP bfloat16')
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            main()
    elif args.precision == "float16":
        print('---- Enable AMP float16')
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            main()
    else:
        main()
