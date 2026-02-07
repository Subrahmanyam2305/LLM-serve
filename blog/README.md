# LLM-serve Blog

A comprehensive guide to LLM inference optimization, built with Jekyll and hosted on GitHub Pages.

## Live Site

Visit: https://subrahmanyam-ai.github.io/LLM-serve/

## Local Development

### Prerequisites
- Ruby 3.0+ 
- Bundler (`gem install bundler`)

### Setup

```bash
cd blog
bundle install
```

### Run Locally

```bash
bundle exec jekyll serve
```

Then open http://localhost:4000/LLM-serve/ in your browser.

### Build for Production

```bash
bundle exec jekyll build
```

Output will be in `_site/` directory.

## Structure

```
blog/
├── _config.yml          # Jekyll configuration
├── _layouts/
│   └── default.html     # Main layout template
├── assets/
│   ├── css/
│   │   └── style.css    # Custom styles
│   └── images/          # Benchmark charts and diagrams
├── index.md             # Main blog content
├── Gemfile              # Ruby dependencies
└── README.md            # This file
```

## Deployment

The blog automatically deploys to GitHub Pages when changes are pushed to the `main` branch. See `.github/workflows/deploy-blog.yml` for the CI/CD configuration.

## Editing the Blog

The entire blog is in `index.md`. It uses:
- Standard Markdown for content
- HTML `<div>` tags for custom styling (callouts, diagrams, etc.)
- ASCII art for diagrams (wrapped in `<div class="ascii-diagram">`)

### Custom CSS Classes

| Class | Purpose |
|-------|---------|
| `.callout` | Info/warning/success boxes |
| `.key-insight` | Highlighted insight boxes |
| `.metric-grid` | Grid of metric cards |
| `.ascii-diagram` | Monospace diagram blocks |
| `.toc` | Table of contents styling |
| `.image-caption` | Caption text under images |

## License

MIT - See main repository LICENSE file.
