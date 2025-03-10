
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def set_sparsity(model, sparsity):
    for module in model.modules():
        if module.__class__.__name__.__contains__("AttentionExperimental"):
            module.token_sparse_method = sparsity
            module.set_token_sparsity()
    return model

def main():
    # question = "A $y$-intercept is a point on the graph that lies on the $y$-axis, so $x = 0$. Hence, the number $y$-intercepts corresponds to the number of real solutions of the quadratic equation $y^2 - 4y - 1 = 0$. The discriminant of this quadratic equation is $(-4)^2 + 4 \cdot 1 \cdot (-1) = 20$, which is positive, so the quadratic has two distinct real roots. Therefore, the number of $y$-intercepts is $\boxed{2}$. \n  \n [asy] \n size(150); \n real ticklen=3; \n real tickspace=2; \n  \n real ticklength=0.1cm; \n real axisarrowsize=0.14cm; \n pen axispen=black+1.3bp; \n real vectorarrowsize=0.2cm; \n real tickdown=-0.5; \n real tickdownlength=-0.15inch; \n real tickdownbase=0.3; \n real wholetickdown=tickdown; \n void rr_cartesian_axes(real xleft, real xright, real ybottom, real ytop, real xstep=1, real ystep=1, bool \n  \n useticks=false, bool complexplane=false, bool usegrid=true) { \n  \n import graph; \n  \n real i; \n  \n if(complexplane) { \n  \n label('$\textnormal{Re}$',(xright,0),SE); \n  \n label('$\textnormal{Im}$',(0,ytop),NW); \n  \n } else { \n  \n label('$x$',(xright+0.4,-0.5)); \n  \n label('$y$',(-0.5,ytop+0.2)); \n  \n } \n  \n ylimits(ybottom,ytop); \n  \n xlimits( xleft, xright); \n  \n real[] TicksArrx,TicksArry; \n  \n for(i=xleft+xstep; i<xright; i+=xstep) { \n  \n if(abs(i) >0.1) { \n  \n TicksArrx.push(i); \n  \n } \n  \n } \n  \n for(i=ybottom+ystep; i<ytop; i+=ystep) { \n  \n if(abs(i) >0.1) { \n  \n TicksArry.push(i); \n  \n } \n  \n } \n  \n if(usegrid) {"    
    question = "If millionaires have butlers, why don't million dollar language models have a butler too? I think its because "    

    # Model name from the Hugging Face Hub
    # model_name = "akhauriyash/Llama-3.2-1B-Butler"                    ### [VALIDATED]
    # model_name = "akhauriyash/Llama-3.2-3B-Butler"                    ### [VALIDATED]
    model_name = "akhauriyash/Llama-3.1-8B-Butler"                    ### [VALIDATED]
    # model_name = "akhauriyash/DeepSeek-R1-Distill-Llama-8B-Butler"      ### [VALIDATED]
    # model_name = "akhauriyash/Llama-2-7b-hf-Butler"                   ### [VALIDATED]
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model = set_sparsity(model, "fixed_50pc")
    # Create a text-generation pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    response = generator(
        question, 
        max_new_tokens=200,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    
    print("Question:\n" + "-"*80)
    print(question)
    print("-"*80 + "\nResponse:\n" + "-"*80)
    print(response[0]['generated_text'][len(question):])

if __name__ == "__main__":
    main()