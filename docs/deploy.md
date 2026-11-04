# Putting the demo page on graph.peterparker.ca

The demo page is `site/`: `index.html` with its script, stylesheet and data inline,
`staticwebapp.config.json` (the host's headers) and `robots.txt`. There is no backend, so
hosting it is a file upload and one DNS record, the same path as the other project pages on
peterparker.ca.

**Every number on the page comes from the committed results.** `consort site` writes all three
files from `results/` and the IBM HI-Small runs. When the results change, run `consort report`,
then `consort site`, and commit `site/` with the tables.

**The content security policy allows only the page's own script and stylesheet**, by their
SHA-256 hashes: no other script, no style attribute, no request to anywhere. The hashes change
whenever the page does, so `consort site` writes the policy from the page it has just written,
and `tests/test_host.py` fails in CI if the committed policy has fallen behind the committed
page. A page whose policy is stale shows nothing but its headings, and says why only in the
browser console.

**Deployed 2026-11-04.** The app is `graph-peterparker-ca` in resource group `rg-portfolio-2`
(subscription `PAP-POCs 2`, East US 2, Free plan), served at
`victorious-moss-0866b5f0f.5.azurestaticapps.net`. The deploy token is the repository secret
`AZURE_STATIC_WEB_APPS_API_TOKEN`, the repository variable `SITE` is `on`, and
`.github/workflows/site.yml` uploads `site/` on every push to `main` that changes it. Before
upload, the page was served locally with the policy's headers and rendered in headless Edge:
every chart drew and the console reported no violation. Cost: the Free plan, CA$0.

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

## 3. The DNS record, at Cloudflare (by hand)

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
