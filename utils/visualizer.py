import wandb
import numpy as np

class Visualizer(object):
    """ Visualizer using Weights & Biases (WandB)
    """
    def __init__(self, project='semantic-segmentation', name=None, config=None, 
                 resume=None, id=None, notes=None, tags=None):
        """
        Initialize WandB visualizer
        
        Args:
            project (str): WandB project name
            name (str): Run name (experiment name)
            config (dict): Configuration dictionary to log
            resume (str): 'allow', 'must', 'never', or None
            id (str): Unique run ID for resuming
            notes (str): Notes about the run
            tags (list): Tags for organizing runs
        """
        self.id = id
        self.project = project
        self.run_name = name
        
        # Initialize WandB
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            resume=resume,
            id=id,
            notes=notes,
            tags=tags
        )
        
        print(f"✓ WandB initialized: {self.run.url}")

    def vis_scalar(self, name, x, y, opts=None):
        """
        Log scalar values to WandB
        
        Args:
            name (str): Metric name
            x (int/float): Step (epoch or iteration)
            y (int/float): Value to log
            opts (dict): Additional options (not used in WandB)
        """
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        # Log each (x, y) pair without explicit step to avoid conflicts
        # WandB will auto-increment steps, and we can rely on commit=False for batching
        for step, value in zip(x, y):
            wandb.log({name: value}, commit=False)

    def vis_image(self, name, img, env=None, opts=None):
        """
        Log image to WandB
        
        Args:
            name (str): Image name
            img (np.array): Image array in CHW format (3, H, W) or HWC format
            env (str): Not used in WandB (kept for compatibility)
            opts (dict): Additional options (not used in WandB)
        """
        # Convert CHW to HWC if needed
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))
        
        # Log image to WandB
        wandb.log({name: wandb.Image(img, caption=name)})
    
    def vis_table(self, name, tbl, step=None, opts=None):
        """
        Log table data to WandB
        
        Args:
            name (str): Table name
            tbl (dict): Dictionary of key-value pairs
            step (int): Step number for logging (optional)
            opts (dict): Additional options (not used in WandB)
        """
        # Log as config if it's the Options table
        if name == "Options" or name == "[Options]":
            wandb.config.update(tbl, allow_val_change=True)
        else:
            # For other tables (like Class IoU), log as a WandB table
            if isinstance(tbl, dict):
                columns = ["Class", "Value"]
                data = [[str(k), float(v) if isinstance(v, (int, float)) else str(v)] 
                        for k, v in tbl.items()]
                table = wandb.Table(columns=columns, data=data)
                if step is not None:
                    wandb.log({name: table}, step=step)
                else:
                    wandb.log({name: table}, commit=False)
            else:
                # If not a dict, just log as is
                if step is not None:
                    wandb.log({name: tbl}, step=step)
                else:
                    wandb.log({name: tbl}, commit=False)
    
    def finish(self):
        """Finish WandB run"""
        wandb.finish()


if __name__=='__main__':
    # Example usage
    vis = Visualizer(project='test-project', name='test-run')
    
    # Test table logging
    tbl = {"lr": 214, "momentum": 0.9}
    vis.vis_table("test_table", tbl)
    
    # Test scalar logging
    vis.vis_scalar(name='loss', x=0, y=1.5)
    vis.vis_scalar(name='loss', x=1, y=1.2)
    vis.vis_scalar(name='loss', x=2, y=0.9)
    
    # Test image logging
    test_img = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    vis.vis_image('test_image', test_img)
    
    vis.finish()