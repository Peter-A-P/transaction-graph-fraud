# Putting the demo page on graph.peterparker.ca

The demo site is `site/`: two pages (`index.html` and `architecture.html`), one stylesheet,
one script (`app.js`), two fonts, the three architecture diagrams, `data.json`, the host's
configuration (`staticwebapp.config.json`) and `robots.txt`. There is no backend, so hosting it
is a file upload and one DNS record, the same path as the other project pages on
peterparker.ca. It is styled like them: the stylesheet starts from theirs, with the same type,
palette, header, menu, figure cards and callouts.

**Every number on the page comes from `site/data.json`**, and none is typed by hand: the
figures, and every number in the findings beside them, are drawn by `app.js` from that file.
`consort site` writes it from the committed results (`results/summary.json`,
`results/faithfulness.json`, `results/serving.json` and the gate's ledger), adds the ring,
which needs IBM's data and the exported model, and copies the diagrams from
`docs/architecture/`. When the results change, run `consort report`, then `consort site`, and
commit `site/` with the tables. In CI, `tests/test_page.py` fails if the committed data.json
has fallen behind the results, and `tests/test_host.py` if a diagram has fallen behind the
documentation's or the script looks for an element the page does not have.

**The film** (`site/video/ring.mp4`, 13.5 seconds, 1080 by 1080, silent, with
`ring-poster.jpg`) is the ring on the page told in four steps: one transfer the row model lets
through, the ring around it, both models scoring it, and the headline figures. It is drawn by
`video/ring.html` from `site/data.json`, so its numbers and its ring are the page's, and
rendered by `video/render.py`, which screenshots each frame in the machine's own Edge and
encodes them with imageio-ffmpeg's bundled ffmpeg:

    uv run --no-project --with playwright --with imageio-ffmpeg python video/render.py

Rerun it after `consort site` whenever the results change; the film is committed, not built in
CI.

**The content security policy allows this origin and nothing else**: no inline script, no
style attribute, no font, script or request from anywhere else. The fonts (Newsreader and
Inter, SIL Open Font License) are served from the site itself for that reason.
`tests/test_host.py` fails if a page uses anything the policy refuses, which would otherwise
show only as a blank chart and a line in the browser console.

**Deployed 2026-11-04.** The app is `graph-peterparker-ca` in resource group `rg-portfolio-2`
(subscription `PAP-POCs 2`, East US 2, Free plan), served at
`victorious-moss-0866b5f0f.5.azurestaticapps.net`. The deploy token is the repository secret
`AZURE_STATIC_WEB_APPS_API_TOKEN`, the repository variable `SITE` is `on`, and
`.github/workflows/site.yml` uploads `site/` on every push to `main` that changes it. Before
upload, the page was served locally with the policy's headers and rendered in headless Edge:
every chart drew and the console reported no violation. The CNAME went in at Cloudflare and
`graph.peterparker.ca` validated and served over HTTPS with Azure's certificate the same day.
Cost: the Free plan, CA$0.

## 1. The app (done)

```powershell
az staticwebapp create --name graph-peterparker-ca --resource-group rg-portfolio-2 `
  --location eastus2 --sku Free --subscription "PAP-POCs 2"
```

Created with no deployment source, so Azure commits no workflow and wires no credential into
the repository; the repository's own workflow does the upload.

## 2. The deploy token and the switch (done)

The token goes from the Azure CLI straight into GitHub's secret store, never on screen, in a
file or in shell history. **The token alone is enough to publish to the site.**

```powershell
az staticwebapp secrets list --name graph-peterparker-ca --resource-group rg-portfolio-2 `
  --subscription "PAP-POCs 2" --query properties.apiKey -o tsv |
  gh secret set AZURE_STATIC_WEB_APPS_API_TOKEN --repo Peter-A-P/transaction-graph-fraud

gh variable set SITE --body on --repo Peter-A-P/transaction-graph-fraud
```

Until `SITE` is `on`, every run of the website workflow skips rather than failing. To publish
without a change: `gh workflow run site.yml --repo Peter-A-P/transaction-graph-fraud`.

## 3. The DNS record, at Cloudflare (by hand, done)

In the Cloudflare dashboard, `peterparker.ca`, **DNS**, **Records**, **Add record**:

| Type | Name | Target | Proxy status | TTL |
|---|---|---|---|---|
| CNAME | `graph` | `victorious-moss-0866b5f0f.5.azurestaticapps.net` | **DNS only** | Auto |

**DNS only, the grey cloud.** A proxied record hides the target behind Cloudflare's addresses,
Azure's validation cannot see the CNAME, and the certificate is never issued. Then:

```powershell
az staticwebapp hostname set --name graph-peterparker-ca --resource-group rg-portfolio-2 `
  --hostname graph.peterparker.ca --subscription "PAP-POCs 2"
```

Validation usually takes minutes; Azure issues and renews the certificate itself.

## 4. Check it

- `https://graph.peterparker.ca` serves over HTTPS with no certificate warning.
- The browser console on the live page is empty: a policy violation is reported there and
  nowhere else.
- The page's footer names the commit it was built from, and its figures match the README.
- Then give the project's entry in peterparker.ca's `projects.yaml` a `demo:` address, so the
  project page carries the button.
