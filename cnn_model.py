import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self,params):
        super(Model, self).__init__()
        # here we assume that input media can only be square
        self.input_size = 28
        default_params ={
            'max_pool_kernel_size':2,
            'max_pool_stride':2,
            'max_pool_padding':0,
            
            'conv1_kernel_size':3,
            'conv1_stride':1,
            'conv1_padding':1,
            'conv1_in_channels':1,
            'conv1_out_channels':32,

            'conv2_kernel_size':3,
            'conv2_stride':1,
            'conv2_padding':1,
            'conv2_in_channels':32,
            'conv2_out_channels':64,

            'conv3_kernel_size':3,
            'conv3_stride':1,
            'conv3_padding':1,
            'conv3_in_channels':64,
            'conv3_out_channels': 256,
    
            'linear1_output':1000,
            'linear2_output':500,
            
            'learning_rate':0.005,
            'momentum':0.5
        }
        default_params.update(params)

        max_pool_param = {
            'kernel_size':default_params['max_pool_kernel_size'],
            'stride':default_params['max_pool_stride'],
            'padding':default_params['max_pool_padding']
            }
        self.pooling = nn.MaxPool2d(**max_pool_param)
        #self.pooling = nn.DataParallel(self.pooling)
        
        self.conv1 = {
            'kernel_size':default_params['conv1_kernel_size'],
            'padding':default_params['conv1_padding'],
            'stride':default_params['conv1_stride'],
            'in_channels':default_params['conv1_in_channels'],
            'out_channels':default_params['conv1_out_channels']
            }
        c1_dim = (self.input_size-self.conv1['kernel_size']+2*self.conv1['padding'])/self.conv1['stride']+1
        c1_pooled_dim = (c1_dim-max_pool_param['kernel_size'])/max_pool_param['stride'] + 1
        self.conv1_layer = nn.Conv2d(**self.conv1)
        #self.conv1_layer = nn.DataParallel(self.conv1_layer)
        
        self.conv2 = {
            'kernel_size':default_params['conv2_kernel_size'],
            'padding':default_params['conv2_padding'],
            'stride':default_params['conv2_stride'],
            'in_channels':default_params['conv2_in_channels'],
            'out_channels':default_params['conv2_out_channels']
            }
        if self.conv2['in_channels'] != self.conv1['out_channels']:
            raise
        c2_dim = (c1_pooled_dim-self.conv2['kernel_size']+2*self.conv2['padding'])/self.conv2['stride']+1
        c2_pooled_dim = (c2_dim-max_pool_param['kernel_size'])/max_pool_param['stride'] + 1
        self.conv2_1_layer = nn.Conv2d(**self.conv2)
        #self.conv2_1_layer = nn.DataParallel(self.conv2_1_layer)
        self.conv2_2_layer = nn.Conv2d(**self.conv2)
        #self.conv2_2_layer = nn.DataParallel(self.conv2_2_layer)
        
        self.conv3 = {
            'kernel_size':default_params['conv3_kernel_size'],
            'padding':default_params['conv3_padding'],
            'stride':default_params['conv3_stride'],
            'in_channels':default_params['conv3_in_channels'],
            'out_channels':default_params['conv3_out_channels']
            }
        if self.conv3['in_channels'] != self.conv2['out_channels']:
            raise
        self.c3_dim = int((c2_pooled_dim-self.conv3['kernel_size']+2*self.conv3['padding'])/self.conv3['stride']+1)
        self.conv3_1_layer = nn.Conv2d(**self.conv3)
        #self.conv3_1_layer = nn.DataParallel(self.conv3_1_layer)
        self.conv3_2_layer = nn.Conv2d(**self.conv3)
        #self.conv3_2_layer = nn.DataParallel(self.conv3_2_layer)
        
        self.linear1 = nn.Linear(int(self.conv3['out_channels']*2*self.c3_dim*self.c3_dim), default_params['linear1_output'])
        #self.linear1 = nn.DataParallel(self.linear1)
        
        self.linear2 = nn.Linear(default_params['linear1_output'], default_params['linear2_output'])
        #self.linear2 = nn.DataParallel(self.linear2)
        
        self.out = nn.Linear(default_params['linear2_output'],10)
        #self.out = nn.DataParallel(self.out)

    def forward(self, x):
        
        c1 = self.pooling(F.relu(self.conv1_layer(x)))

        c2_1 = self.pooling(F.relu(self.conv2_1_layer(c1)))
        c2_2 = self.pooling(F.relu(self.conv2_2_layer(c1)))

        
        c3_1 = F.relu(self.conv3_1_layer(c2_1))
        c3_2 = F.relu(self.conv3_1_layer(c2_2))

        
        c4 = torch.cat([c3_1,c3_2],1)

        c5 = c4.view(-1,self.conv3['out_channels']*2*self.c3_dim*self.c3_dim)

        
        l1 = F.relu(self.linear1(c5))

        
        l2 = F.relu(self.linear2(l1))

        o = self.out(l2)
        so = F.log_softmax(o)
        return so

def train_model(model,epoch,optimizer,train_loader, log_interval=10, use_cuda = False):
    """
    input: model object
    input: epoch (used during printing)
    input: optimizer
    input: train_loader, iterable on tuples (data, label)
    return: void
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda and torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
                epoch, batch_idx*len(data),len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            torch.save(model.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

def eval_model(model,test_loader, use_cuda = False):
    """
    input: test_loader, iterable on tuples (data, label)
    return: dict with metrics
    """
    metrics = {}
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda and torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        metrics['avg_loss'] = test_loss
        metrics['accuracy'] = float(correct)/float(len(test_loader.dataset))
    return metrics

def default_optimizer(model, learning_rate = 0.005, momentum=0.5):
    return optim.SGD(model.parameters(), learning_rate,momentum)

def run_train_and_eval(model, optimizer, n_epochs, test_loader, train_loader, use_cuda =False):
    """
    input: model
    input: optimizer
    input: n_epochs, number of passes on training_data
    input: test_loader, iterable on tuples (data, label)
    input: train_loader, iterable on tuples (data, label)
    return: dict with test metrics
    """
    eval_model(model,test_loader, use_cuda)
    test_metrics = None
    for epoch in range(1, n_epochs + 1):
        train_model(model, epoch, optimizer, train_loader, use_cuda=use_cuda)
    test_metrics = eval_model(model, test_loader, use_cuda)
    return test_metrics

def hyperpameter_tuning(test_loader, train_loader, parameters, metric, maximize = True):
    """
    input: test_loader
    input: train_loader
    input: parameters, dict containing values, ranges or lists for parameters
    input: metric, string representing a test metric to optimizer, currently needs to be in [accuracy, avg_loss]
    input: maximize, whether we want to maximize the metric

    return: (best metric found, parameters that generated best metric)
    """
    parameters_list = generate_parameters(parameters)
    best_so_far = None
    best_metric_so_far = None
    for parameters in parameters_list:
        print("current parameters: ", parameters)
        nm = Model(parameters)
        use_cuda = torch.cuda.is_available()
        if use_cuda and torch.cuda.device_count() > 1:
            nm = nn.DataParallel(nm)
        if use_cuda:
            nm.cuda()
        optimizer = default_optimizer(nm, parameters['learning_rate'],parameters['momentum'])
        metrics = run_train_and_eval(nm, optimizer, 1, test_loader, train_loader, use_cuda)
        if best_metric_so_far is None or (not maximize and best_metric_so_far > metrics[metric]) or (maximize and best_metric_so_far < metrics[metric]):
            best_metric_so_far = metrics[metric]
            best_so_far = parameters
    return (best_metric_so_far, best_so_far)


def generate_parameters(parameters):
    if len(parameters) == 0:
        return []

    out = []
    print(parameters)
    for k in parameters:
        nout = []
        vals = parameters[k]
        if isinstance(vals, (int, float, complex)):
            vals = [vals]
        if len(out) == 0:
            for v in vals:
                out.append({k:v})
        else:
            for o in out:
                for v in vals:
                    no = o.copy()
                    no[k] = v
                    nout.append(no)
            out = nout
    return out
