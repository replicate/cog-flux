# cog-flux

This is a [Cog](https://cog.run) inference model for FLUX.1 [schnell] and FLUX.1 [dev] by [Black Forest Labs](https://blackforestlabs.ai/). It powers the following Replicate models:

* https://replicate.com/black-forest-labs/flux-schnell
* https://replicate.com/black-forest-labs/flux-dev

## Features

* Compilation with `torch.compile`
* Optional fp8 quantization based on [aredden/flux-fp8-api](https://github.com/aredden/flux-fp8-api), using fast CuDNN attention from Pytorch nightlies
* NSFW checking with [CompVis](https://huggingface.co/CompVis/stable-diffusion-safety-checker) and [Falcons.ai](https://huggingface.co/Falconsai/nsfw_image_detection) safety checkers
* img2img support

## Getting started

If you just want to use the models, you can run [FLUX.1 [schnell]](https://replicate.com/black-forest-labs/flux-schnell) and [FLUX.1 [dev]](https://replicate.com/black-forest-labs/flux-dev) on Replicate with an API or in the browser.

The code in this repo can be used as a template for customizations on FLUX.1, or to run the models on your own hardware.

First you need to select which model to run:

```shell
script/select.sh {dev,schnell}
```

Then you can run a single prediction on the model using:

```shell
cog predict -i prompt="a cat in a hat"
```

The [Cog getting started guide](https://cog.run/getting-started/) explains what Cog is and how it works.

To deploy it to Replicate, run:

```shell
cog login
cog push r8.im/<your-username>/<your-model-name>
```

Learn more on [the deploy a custom model guide in the Replicate documentation](https://replicate.com/docs/guides/deploy-a-custom-model).

## Contributing

Pull requests and issues are welcome! If you see a novel technique or feature you think will make FLUX.1 inference better or faster, let us know and we'll do our best to integrate it.

## Rough, partial roadmap

* Serialize quantized model instead of quantizing on the fly
* Use row-wise quantization
* Port quantization and compilation code over to https://github.com/replicate/flux-fine-tuner

## License

The code in this repository is licensed under the [Apache-2.0 License](LICENSE).

FLUX.1 [dev] falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

FLUX.1 [schnell] falls under the [Apache-2.0 License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).
