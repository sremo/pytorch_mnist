import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self,params):
        super(Model, self).__init__()
        
        # here we assume that input media can only be square
        self.input_size = 28
        
        max_pool_param = {
            'kernel_size':params['max_pool_kernel_size'],
            'stride':params['max_pool_stride'],
            'padding':params['max_pool_padding']
            }
        self.pooling = nn.MaxPool2d(**max_pool_param)
        
        self.conv1 = {
            'kernel_size':params['conv1_kernel_size'],
            'padding':params['conv1_padding'],
            'stride':params['conv1_stride'],
            'in_channels':params['conv1_in_channels'],
            'out_channels':params['conv1_out_channels']
            }

        c1_dim = (self.input_size-self.conv1['kernel_size']+2*self.conv1['padding'])/self.conv1['stride']+1
        c1_pooled_dim = (c1_dim-max_pool_param['kernel_size'])/max_pool_param['stride'] + 1
        self.conv1_layer = nn.Conv2d(**self.conv1)
        
        self.conv2 = {
            'kernel_size':params['conv2_kernel_size'],
            'padding':params['conv2_padding'],
            'stride':params['conv2_stride'],
            'in_channels':params['conv2_in_channels'],
            'out_channels':params['conv2_out_channels']
            }

        if self.conv2['in_channels'] != self.conv1['out_channels']:
            raise

        
        c2_dim = (c1_pooled_dim-self.conv2['kernel_size']+2*self.conv2['padding'])/self.conv2['stride']+1
        c2_pooled_dim = (c2_dim-max_pool_param['kernel_size'])/max_pool_param['stride'] + 1
        self.conv2_1_layer = nn.Conv2d(**self.conv2)
        self.conv2_2_layer = nn.Conv2d(**self.conv2)
        
        
        self.conv3 = {
            'kernel_size':params['conv3_kernel_size'],
            'padding':params['conv3_padding'],
            'stride':params['conv3_stride'],
            'in_channels':params['conv3_in_channels'],
            'out_channels':params['conv3_out_channels']
            }

        if self.conv3['in_channels'] != self.conv2['out_channels']:
            raise

        self.c3_dim = int((c2_pooled_dim-self.conv3['kernel_size']+2*self.conv3['padding'])/self.conv3['stride']+1)
        self.conv3_1_layer = nn.Conv2d(**self.conv3)
        self.conv3_2_layer = nn.Conv2d(**self.conv3)
        

        self.linear1 = nn.Linear(int(self.conv3['out_channels']*2*self.c3_dim*self.c3_dim), params['linear1_output'])
        

        self.linear2 = nn.Linear(params['linear1_output'], params['linear2_output'])
        self.out = nn.Linear(params['linear2_output'],10)
        

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
        return F.log_softmax(o)

def train_model(epoch,model,optimizer,train_loader):
    log_interval = 1000
    model.train()
    max_ct = 3
    ct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if ct > max_ct:
            break
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            ct = ct + 1
            print("Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
                    epoch, batch_idx*len(data),len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            torch.save(model.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test_model(model,test_loader):
    """
    input: test_loader
    returns: dict with metrics
    """
    metrics = {}
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    metrics['avg_loss'] = test_loss
    metrics['accuracy'] = correct/len(test_loader.dataset)


def default_optimizer(model, learning_rate = 0.005, momentum=0.5):
    return optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)


def run_train_and_eval(model, optimizer, n_epochs, test_loader, train_loader):
    n_epochs = 1
    test_model(model, test_loader)
    test_metrics = None
    for epoch in range(1, n_epochs + 1):
        train_model(epoch, model, optimizer, train_loader)
        test_metrics = test_model(model, test_loader)
    return test_metrics

def hyperpameter_tuning(test_loader, train_loader, metric, parameters):
    """
    parameters contains ranges/ lists for parameters to tune and default values for parameters to keep steady
    metric specifies what metric to optimize on

    returns best parameters set with best metric
    can print metrics for various parameters
    """
    parameters_list = generate_parameters(parameters)
    best_so_far = None
    best_metric_so_far = None
    for parameters in parameters_list:
        # model = create model
        nm = Model(parameters)
        optimizer = default_optimizer(nm, parameters['learning_rate'],parameters['momentum'])
        metrics = run_train_and_eval(nm, optimizer, 2, test_loader, train_loader)
        if best_metric_so_far is None or best_metric_so_far < metrics[metric]:
            best_metric_so_far = metrics[metric]
            best_so_far = parameters
    return (best_metric_so_far, best_so_far)


def generate_parameters(parameters):
    if len(parameters) == 0:
        return []

    out = []
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
