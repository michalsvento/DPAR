import os
from tqdm import tqdm
import torch
import torchaudio
import numpy as np

import utils.blind_bwe_utils as blind_bwe_utils
import wandb
import omegaconf
import utils.logging as utils_logging
import librosa
import wandb
import utils.add_desired_sdr as add_sdr 
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import matplotlib.pyplot as plt

class BlindSampler():

    def __init__(self, model,  diff_params, args):

        self.model = model

        self.diff_params = diff_params #same as training, useful if we need to apply a wrapper or something
        self.args=args
        if not(self.args.tester.diff_params.same_as_training):
            self.update_diff_params()

        self.order=self.args.tester.order

        self.xi=self.args.tester.posterior_sampling.xi
        #hyperparameter for the reconstruction guidance
        self.data_consistency=self.args.tester.posterior_sampling.data_consistency #use reconstruction gudance without replacement
        self.nb_steps=self.args.tester.T
        self.dpir = self.args.dpir

        self.start_sigma=self.args.tester.posterior_sampling.start_sigma
        if self.start_sigma =="None":
            self.start_sigma=None

        print("start sigma", self.start_sigma)

        self.operator=None

        def loss_fn_rec(x_hat, x):
                diff=x_hat-x
                #if self.args.tester.filter_out_cqt_DC_Nyq:
                #    diff=self.model.CQTransform.apply_hpf_DC(diff)
                return (diff**2).sum()/2

        self.rec_distance=lambda x_hat, x: loss_fn_rec(x_hat, x)


    def update_diff_params(self):
        #the parameters for testing might not be necesarily the same as the ones used for training
        self.diff_params.sigma_min=self.args.tester.diff_params.sigma_min
        self.diff_params.sigma_max =self.args.tester.diff_params.sigma_max
        self.diff_params.ro=self.args.tester.diff_params.ro
        self.diff_params.sigma_data=self.args.tester.diff_params.sigma_data
        self.diff_params.Schurn=self.args.tester.diff_params.Schurn
        self.diff_params.Stmin=self.args.tester.diff_params.Stmin
        self.diff_params.Stmax=self.args.tester.diff_params.Stmax
        self.diff_params.Snoise=self.args.tester.diff_params.Snoise


    def get_rec_grads(self, x_hat, y, x, t_i):
        """
        Compute the gradients of the reconstruction error with respect to the input
        """ 

        if self.args.tester.posterior_sampling.annealing_y.use:
            if self.args.tester.posterior_sampling.annealing_y.mode=="same_as_x":
                y=y+torch.randn_like(y)*t_i
            elif self.args.tester.posterior_sampling.annealing_y.mode=="same_as_x_limited":
                t_min=torch.Tensor([self.args.tester.posterior_sampling.annealing_y.sigma_min]).to(y.device)
                #print(t_i, t_min)
                t_y=torch.max(t_i, t_min)
                #print(t_y)

                y=y+torch.randn_like(y)*t_y
            elif self.args.tester.posterior_sampling.annealing_y.mode=="fixed":
                t_min=torch.Tensor([self.args.tester.posterior_sampling.annealing_y.sigma_min]).to(y.device)
                #print(t_i, t_min)

                y=y+torch.randn_like(y)*t_min

        print("y",y.std(), "x_hat",x_hat.std())
        norm=self.rec_distance(self.operator.degradation(x_hat), y)
        print("norm:", norm.item())

        rec_grads=torch.autograd.grad(outputs=norm.sum(),
                                      inputs=x)

        rec_grads=rec_grads[0]
        
        if self.args.tester.posterior_sampling.normalization=="grad_norm":
        
            normguide=torch.norm(rec_grads)/self.args.exp.audio_len**0.5
            #normguide=norm/self.args.exp.audio_len**0.5
        
            #normalize scaling
            s=self.xi/(normguide+1e-6)

        elif self.args.tester.posterior_sampling.normalization=="loss_norm":
            normguide=norm/self.args.exp.audio_len**0.5
            #normguide=norm/self.args.exp.audio_len**0.5
        
            #normalize scaling
            s=self.xi/(normguide+1e-6)
        
        #optionally apply a treshold to the gradients
        if False:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        
        return s*rec_grads/t_i, norm

    def get_score_rec_guidance(self, x, y, t_i):

        x.requires_grad_()
        x_hat=self.get_denoised_estimate(x, t_i)

        rec_grads=self.get_rec_grads(x_hat, y, x, t_i)

        
        score=self.denoised2score(x_hat, x, t_i)

        score=score-rec_grads

        return score
    
    def get_denoised_estimate(self, x, t_i):
        x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))

        if self.args.tester.filter_out_cqt_DC_Nyq:
            x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)
        return x_hat
    

    def get_score(self,x, y, t_i):
        if y==None:
            assert self.operator==None
            #unconditional sampling
            with torch.no_grad():
                x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                if self.args.tester.filter_out_cqt_DC_Nyq:
                    x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)
                score=(x_hat-x)/t_i**2
            return score
        else:
            if self.xi>0:
                assert self.operator is not None
                #apply rec. guidance
                score=self.get_score_rec_guidance(x, y, t_i)
    
                #optionally apply replacement or consistency step
                if self.data_consistency:
                    raise NotImplementedError
                    #convert score to denoised estimate using Tweedie's formula
                    x_hat=score*t_i**2+x
    
                    try:
                        x_hat=self.data_consistency_step(x_hat)
                    except:
                        x_hat=self.data_consistency_step(x_hat,y, degradation)
    
                    #convert back to score
                    score=(x_hat-x)/t_i**2
    
            else:
                #raise NotImplementedError
                #denoised with replacement method
                with torch.no_grad():
                    x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                        
                    #x_hat=self.data_consistency_step(x_hat,y, degradation)
                    if self.data_consistency:
                        raise NotImplementedError
                        try:
                            x_hat=self.data_consistency_step(x_hat)
                        except:
                            try:
                                x_hat=self.data_consistency_step(x_hat,y, degradation)
                            except:
                                x_hat=self.data_consistency_step(x_hat,y, degradation, filter_params)

        
                    score=(x_hat-x)/t_i**2
    
            return score



    def resample(self,x):
        N=100
        return torchaudio.functional.resample(x,orig_freq=int(N*self.factor), new_freq=N)


    def prepare_smooth_mask(self, mask, size=10):
        hann=torch.hann_window(size*2)
        hann_left=hann[0:size]
        hann_right=hann[size::]
        B,N=mask.shape
        mask=mask[0]
        prev=1
        new_mask=mask.clone()
        #print(hann.shape)
        for i in range(len(mask)):
            if mask[i] != prev:
                #print(i, mask.shape, mask[i], prev)
                #transition
                if mask[i]==0:
                   print("apply right")
                   #gap encountered, apply hann right before
                   new_mask[i-size:i]=hann_right
                if mask[i]==1:
                   print("apply left")
                   #gap encountered, apply hann left after
                   new_mask[i:i+size]=hann_left
                #print(mask[i-2*size:i+2*size])
                #print(new_mask[i-2*size:i+2*size])
                
            prev=mask[i]
        return new_mask.unsqueeze(0).expand(B,-1)



        

    def predict_unconditional(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device,
    ):
        self.y=None
        self.degradation=None
        self.reference=None
        #self.args.tester.posterior_sampling.xi=0    #just in case
        #self.args.tester.posterior_sampling.start_sigma="None"    #just in case
        self.start_sigma=None
        self.xi=0
        return self.predict(shape=shape, device=device, conditional=False)

    def predict_dpir(
            self,
            shape,
            device):
        self.y=None
        self.degradation=None
        self.reference=None
        #self.args.tester.posterior_sampling.xi=0    #just in case
        #self.args.tester.posterior_sampling.start_sigma="None"    #just in case
        self.start_sigma=None
        self.xi=0
        return self.predict_pnp(shape=shape, device=device, blind=False, conditional=False)
        

    def predict_conditional(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        x_init=None
    ):
        self.y=y
        return self.predict(shape=y.shape,device=y.device,  blind=False, x_init=x_init)

  
    def denoised2score(self,  x_d0, x, t):
        #tweedie's score function
        return (x_d0-x)/t**2
    def score2denoised(self, score, x, t):
        return score*t**2+x

    def move_timestep(self, x, t, gamma, Snoise=1):
        #if gamma_sig[i]==0 this is a deterministic step, make sure it doed not crash
        t_hat=t+gamma*t
        #sample noise, Snoise is 1 by default
        epsilon=torch.randn(x.shape).to(x.device)*Snoise
        #add extra noise
        x_hat=x+((t_hat**2 - t**2)**(1/2))*epsilon
        return x_hat, t_hat

    def get_alpha_k(self, lambda_, sigma_image, sigma_denoiser):
        alpha = lambda_ * (sigma_image**2/sigma_denoiser**2)
        return alpha

    def add_desired_snr(self, y_clean, snr_dB, n):
        # std = torch.linalg.vector_norm(y_clean, ord=2) / (torch.sqrt(torch.tensor(torch.numel(y_clean)*(10**(snr_dB/10)))))
        std = torch.linalg.vector_norm(y_clean, ord=2) / (torch.linalg.norm(n, ord=2) * torch.sqrt(torch.tensor(10 ** (snr_dB / 10))))
        return std
    
    def add_desired_gap(self, signal, mode='random', threshold=50, gap_length=100, gap_position=None):
        mask = torch.ones_like(signal)
        signal_with_gap = signal.clone()
        if mode == 'random':
            random_var = torch.rand_like(signal)
            mask[random_var < (threshold / 100)] = 0
            signal_with_gap = signal * mask
        if mode == 'fixed_segments':
            for i in range(len(gap_position)):
                mask[gap_position[i]:gap_position[i] + gap_length] = 0
            signal_with_gap = signal * mask

        return signal_with_gap, mask
    



    def step(self, x, t_i, t_i_1, gamma_i, blind=False, y=None): 

        if self.args.tester.posterior_sampling.SNR_observations !="None":
            snr=10**(self.args.tester.posterior_sampling.SNR_observations/10)
            sigma2_s=torch.var(y, -1) 
            sigma=torch.sqrt(sigma2_s/snr).unsqueeze(-1)
            #sigma=torch.tensor([self.args.tester.posterior_sampling.sigma_observations]).unsqueeze(-1).to(y.device)
            #print(y.shape, sigma.shape)
            y=y+sigma*torch.randn(y.shape).to(y.device)
            

        x_hat, t_hat=self.move_timestep(x, t_i, gamma_i, self.diff_params.Snoise)

        x_hat.requires_grad_(True)

        x_den=self.get_denoised_estimate(x_hat, t_hat)

        x_den_2=x_den.clone().detach()

        if blind:
            self.fit_params(x_den_2, y)

        if self.args.tester.posterior_sampling.xi>0 and y is not None:
            rec_grads, rec_loss=self.get_rec_grads(x_den, y, x_hat, t_hat)
        else:
            #adding this so that the code does not crash
            rec_loss=0
            rec_grads=0

        x_hat.detach_()
        uncond_score=self.denoised2score(x_den_2, x_hat, t_hat)

        score=uncond_score-rec_grads

        d=-t_hat*score
        #apply second order correction
        h=t_i_1-t_hat


        if t_i_1!=0 and self.order==2:  #always except last step
            #second order correction2
            #h=t[i+1]-t_hat
            t_prime=t_i_1
            x_prime=x_hat+h*d
            x_prime.requires_grad_(True)

            x_den=self.get_denoised_estimate(x_prime, t_prime)

            x_den_2=x_den.clone().detach()


            if self.xi>0 and y is not None:
                rec_grads, rec_loss =self.get_rec_grads(x_den, y, x_prime, t_prime)
            else:
                rec_loss=0
                rec_grads=0

            x_prime.detach_()

            uncond_score=self.denoised2score(x_den_2, x_prime, t_prime)
            
            score=uncond_score-rec_grads


            d_prime=-t_prime*score

            x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

        elif self.order==1: #first condition  is to avoid dividing by 0
            #first order Euler step
            x=x_hat+h*d

        return x, x_den_2, rec_loss, score, rec_grads
        

    def predict_pnp(
        self,
        shape=None,  #observations (lowpssed signal) Tensor with shape (L,)
        device=None,
        blind=False,
        x_init=None,
        conditional=True,
        ):
        if not conditional:
            self.y=None

        if self.args.tester.wandb.use:
            self.setup_wandb()


        #get shape and device from the observations tensor
        if shape is None:
            shape=self.y.shape
            device=self.y.device

        #initialization
        if self.start_sigma is None:
            t=self.diff_params.create_schedule(self.nb_steps).to(device)
            z=self.diff_params.sample_prior(shape, t[0]).to(device)
        # else:
        #     #get the noise schedule
        #     t = self.diff_params.create_schedule_from_initial_t(self.start_sigma,self.nb_steps).to(self.y.device)
        #     #sample from gaussian distribution with sigma_max variance
        #     if x_init is not None:
        #         x = x_init.to(device) + self.diff_params.sample_prior(shape,t[0]).to(device)
        #     else:
        #         print("using y as warm init")
        #         x = self.y + self.diff_params.sample_prior(shape,t[0]).to(device)

        gamma=self.diff_params.get_gamma(t).to(device)

        # if self.args.tester.wandb.use:
        #     self.wandb_run.log({"y": wandb.Audio(self.args.dset.sigma_data*self.y.squeeze().detach().cpu().numpy(), caption="y", sample_rate=self.args.exp.sample_rate)}, step=0, commit=False) 

        #     spec_sample=utils_logging.plot_spectrogram_from_raw_audio(self.args.dset.sigma_data*self.y, self.args.logging.stft)
        #     self.wandb_run.log({"spec_y": spec_sample}, step=0, commit=False)

        # if self.reference is not None:
        #     if self.args.tester.wandb.use:
        #         self.wandb_run.log({"reference": wandb.Audio(self.args.dset.sigma_data*self.reference.squeeze().detach().cpu().numpy(), caption="reference", sample_rate=self.args.exp.sample_rate)}, step=0, commit=False)

        #         spec_sample=utils_logging.plot_spectrogram_from_raw_audio(self.args.dset.sigma_data*self.reference, self.args.logging.stft)
        #         self.wandb_run.log({"spec_reference": spec_sample}, step=0, commit=True)

        # signal loading
        x, sr = librosa.load('/home/svento/projects/BABE2-music-restoration/test_examples/piano/a60_piano.wav')
        print("Sucessfuly loaded audio")
        print(x.dtype)
        if len(x)<184184:
            x = np.concatenate((x, np.zeros(184184-len(x))), axis=None, dtype=x.dtype)
            print('Audio was extended')
        print("x shape",x.shape, 'x type', x.dtype)
        x = x[:184184]
        
        x = torch.tensor(x)
        print("sr: ", sr)


        # noise vector
        # n, _ = librosa.load('/home/svento/projects/BABE2-music-restoration/test_examples/caruso/denoised_vocals_cropped.wav')
        # n = torch.tensor(n[:184184])
        n = torch.rand_like(x)


        # INPAINTING
        # _, mask = self.add_desired_gap(x, mode='random', threshold=80)
        # print("Mask was created, working", torch.sum(mask)/len(mask))    
        
        # DECLIPPING
        y, mask_2, threshold, percentage = add_sdr.clip_sdr(x.numpy(), self.dpir.declipping.sdr)
        y, mask_2 = torch.tensor(y), torch.tensor(mask_2).to(device)
        print("Mask was created, percentage of clipped: ", percentage)   

        mask = torch.ones_like(x) 


        # DENOISING
        std = self.add_desired_snr(x, self.dpir.snr_dB, n)
        # std = 0.030
        # y = mask * x  + std* n
        y = y.to(device)
        print(type(mask), type(y))
        mask_rep = mask.repeat(self.nb_steps+1,1).T.to(device)
        
        # self.dpir.sigma_noise= std
        print("STD: ", std)
        # Alphas - crucial parameter for mixing generative and 
        alphas = self.get_alpha_k(self.dpir.lambda_, std, t)
        alphas = alphas.to(device)
        
        # Consistency and Generative term
        q = mask_rep+alphas.unsqueeze(0)
        consistency_term = torch.div(mask_rep, q).to(device)
        generative_term = torch.div(alphas.unsqueeze(0), q).to(device)
        print(consistency_term.shape, generative_term.shape, y.shape, x.shape)
        x0 = x.clone()
        # y = y * (x0.max() * 0.30 /threshold)

        for i in tqdm(range(0, self.nb_steps, 1)):
            self.step_count=i
            x = consistency_term[...,i] * y + generative_term[...,i] * z
            # if i>=46:
            #     x = x * energy 
            out=self.step(x, t[i], t[i+1], gamma[i], blind=blind, y=None)
            z, x_den, rec_loss, score, lh_score =out
            energy = torch.linalg.norm(mask_2*y) / torch.linalg.norm(mask_2*x) 
            snr = scale_invariant_signal_noise_ratio(x.squeeze(0).to(device), x0.to(device))
            wandb.log({"si-snr": snr, "energy": energy})
        
        # x = (mask_2) * y + (1-mask_2)*x

        x = energy* x



        timesteps= np.arange(0,184184) * (1/float(sr))
        # Create traces for each waveform

        # Create Matplotlib figure and plot
        plt.figure(figsize=(8, 6))
        plt.plot(timesteps, y.detach().cpu().squeeze(0).numpy(), label='Waveform damaged')
        plt.plot(timesteps, x.detach().cpu().squeeze(0).numpy(), label='Waveform repaired')
        plt.xlabel('time')
        plt.ylabel('values')
        plt.title('Two Waveforms')
        plt.legend()

        # Log Matplotlib figure to wandb
        wandb.log({"Two Waveforms": plt})

        if self.args.tester.wandb.use:
            self.wandb_run.finish()

        if blind:
            return x.detach()  , self.operator.params
        else:
            return x.detach() , y.detach(), x0.detach(), snr.detach().cpu().numpy()


        

    def setup_wandb(self):
        config=omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run=wandb.init(project=self.args.tester.wandb.project, entity=self.args.tester.wandb.entity, config=config)
        if self.operator.__class__.__name__== "WaveNetOperator":
            wandb.watch(self.operator.wavenet, log_freq=self.args.tester.wandb.heavy_log_interval, log="all") #wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name=self.args.tester.wandb.run_name +os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id
        #adding the experiment number to the run name, bery important, I hope this does not crash
        self.use_wandb=True

