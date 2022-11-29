# %%
from solu_project.solu_utils import *

model = EasyTransformer("facebook/opt-6.7b")

# %%
text = """ as previously described \[[@B33-microorganisms-06-00037]\]. One unit of trypsin activity was arbitrarily defined as the increase of 0.01 absorbance units at 410 nm. One inhibitor unit was defined as the amount of inhibitor that inhibited one unit of trypsin activity. The quantification of total protein in the fermentation broth, after 48 h of induction, was measured with the Bradford method \[[@B34-microorganisms-06-00037]\], with BSA (Sigma-Aldrich) ranging from 0.781 to 500 µg/mL as standard. Experiments were performed in triplicates.

2.7. Bacterial Strain Used to Biofilm Assay {#sec2dot7-microorganisms-06-00037}
-------------------------------------------

Four gram-positive bacteria methicillin-susceptible *Staphylococcus aureus* MSSA ATCC80958, methicillin-resistant *Staphylococcus aureus* MRSA ATCC33591, *Staphylococcus epidermidis* ATCC 35984 and S. *epidermidis* ATCC 12228 (a non-slime production strain) were used in the biofilm assays. The *S. epidermidis,* known as slime-production INCQ 00650 ATCC 35984 (RP62A), was provided by the Coleção de Microrganismos de Referência em Vigilância Sanitária (CMRVS), FIOCRUZ-INCQS, Rio de Janeiro, RJ, Brazil.

2.8. Biofilm Formation Assay and Determination of Minimal Biofilm Inhibitory Concentration (MBIC) {#sec2dot8-microorganisms-06-00037}
-------------------------------------------------------------------------------------------------

The four bacterial strains were grown in Mueller Hinton agar overnight at 36.5 °C, and a bacterial suspension in 0.9% NaCl, corresponding to 0.5 McFarland scale (1.5 × 10^8^ CFU/mL), was used. An in vitro microtiter plate-based model was performed as described for Staphylococci \[[@B35-microorganisms-06-00037]\], with modifications. Briefly, 20 μL of each bacterial suspension equivalent to the McFarland 0.5 turbidity standard was inoculated onto 96-well microtiter plates with 170 μL of brain heart infusion (BHI) liquid medium. Subsequently, 10 µL of ILTI (previously purified from *Inga laurina* seeds) or GSrILTI fermented broth (rILTI) were added at different concentrations (range from 10 to 1000 µg/mL) to complete 200 μL of final volume per plate well, and the plate was incubated at 36.5 °C for 18 h. In positive controls, 10 μL of sterile distilled water was added instead of ILTI or rILTI. After incubation, the medium was removed, and the wells were washed three times with sterile distilled water. The remaining attached bacteria were fixed with 150 μL of methanol for 20 min. The adherent biofilm layer formed was stained with 0.5% crystal violet for 15 min at room temperature. Then, the dye was removed and this was followed by three washes with sterile distilled water. The preparations were then detained with 200 μL of 95% ethanol for 30 min. Finally, the optical density (OD) of the ethanol-dye suspension was measured at 450 nm. All the strains were tested in triplicate, and the average value for each sample was calculated. Values of higher than 100% represent a stimulation of biofilm formation in comparison with the positive control sample (untreated), in which the well was replaced by sterile distilled water. Vancomycin (20 µg/mL) was used as a negative control of bacteria growth and *S. epidermidis* (ATCC 12228) strain as negative control of biofilm formation. The minimal biofilm inhibitory concentration (MBIC) was defined as minimal concentration at which there were no observable adherent cells in wells stained with crystal violet, according to the approach described above. The concentrations of vancomycin in a range of 15 to 30 µg/mL (MBICs) to each isolates was used as the biofilm formation control.

2.9. Biofilm Detachment Assay and Determination of Minimal Biofilm Eradication Concentration (MBEC) {#sec2dot9-microorganisms-06-00037}
---------------------------------------------------------------------------------------------------

For preformed biofilm disassembly, the mature biofilm was first allowed to accumulate without any supplementation. Briefly, 20 μL of *S. epidermidis* (ATCC 35984) suspension was inoculated onto 96-well microtiter plates with 170 μL of BHI liquid medium at 36.5 °C for 24 h. The ILTI and rILTI influence under biofilm disassembly were pre-established with 24 h old biofilms; then, after this period, biofilms were treated with a range of each ILTI or GSrILTI fermentation broth, containing the recombinant ILTI, concentrations (10 to 100 µg/mL) and incubated at 36.5 °C for an additional 18 h. The amount of residual"""

torch.set_grad_enabled(False)
cache = {}
model.cache_all(cache)
logits = model(model.to_tokens(text))
resids = torch.stack(
    [cache["blocks.{l}.hook_resid_pre"][0] for l in range(model.cfg["n_layers"])],
    axis=0,
)
px.histogram(to_numpy(resids.abs().max([0, 1]).values)).show()

# %%
