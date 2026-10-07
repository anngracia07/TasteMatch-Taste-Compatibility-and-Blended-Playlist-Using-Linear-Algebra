(() => {
  const data = window.PROJECT_DATA;
  if (!data) return;
  const selected = { a: new Set(), b: new Set() };
  let mode = 'MANUAL';
  const byId = new Map(data.songs.map(song => [Number(song.track_id), song]));
  const el = id => document.getElementById(id);
  const label = song => `${song.track_name} — ${song.artists}`;
  const SVG = 'http://www.w3.org/2000/svg';
  const text = (tag, className, value) => { const node = document.createElement(tag); if (className) node.className = className; node.textContent = value; return node; };
  const svg = (tag, attrs = {}) => { const node = document.createElementNS(SVG, tag); Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value))); return node; };

  function renderUser(user) {
    const search = el(`search-${user}`).value.trim().toLowerCase();
    const genre = el(`genre-${user}`).value;
    const list = el(`list-${user}`); list.replaceChildren();
    data.songs.filter(song => (!genre || song.track_genre === genre) && (!search || `${song.track_name} ${song.artists}`.toLowerCase().includes(search))).forEach(song => {
      const row = text('label', 'song-row', '');
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = selected[user].has(Number(song.track_id));
      checkbox.setAttribute('aria-label', `Select ${label(song)} for User ${user.toUpperCase()}`);
      checkbox.addEventListener('change', () => {
        const id = Number(song.track_id);
        if (checkbox.checked && selected[user === 'a' ? 'b' : 'a'].has(id)) {
          checkbox.checked = false;
          showWarning(`${song.track_name} is already selected for the other user. Choose a different song.`);
          return;
        }
        if (checkbox.checked && selected[user].size >= data.maxSongs) {
          checkbox.checked = false;
          showWarning(`User ${user.toUpperCase()} already has ${data.maxSongs} songs. Remove one before adding another.`);
          return;
        }
        checkbox.checked ? selected[user].add(id) : selected[user].delete(id);
        renderUser(user); updateWarning();
      });
      row.append(checkbox, text('span', '', song.track_name), text('small', '', `— ${song.artists} · ${song.track_genre}`));
      list.append(row);
    });
    el(`count-${user}`).textContent = `${selected[user].size} / ${data.maxSongs}`;
    el(`selected-count-${user}`).textContent = `(${selected[user].size})`;
    const chips = el(`selected-${user}`); chips.replaceChildren();
    if (!selected[user].size) chips.append(text('span', 'muted', 'None selected'));
    [...selected[user]].forEach(id => {
      const song = byId.get(id); if (!song) return;
      const chip = text('button', 'chip', `× ${label(song)}`); chip.type = 'button'; chip.title = 'Remove this song';
      chip.addEventListener('click', () => { selected[user].delete(id); renderUser(user); updateWarning(); }); chips.append(chip);
    });
  }
  function showWarning(message) { const warning = el('selection-warning'); warning.textContent = message; warning.classList.remove('hidden'); }
  function updateWarning() {
    if (mode !== 'MANUAL') { el('selection-warning').classList.add('hidden'); return; }
    if (selected.a.size < 5 || selected.b.size < 5) showWarning(`Select at least 5 songs for each user. Current: A ${selected.a.size}, B ${selected.b.size}.`);
    else el('selection-warning').classList.add('hidden');
  }
  function showAutomatic() {
    for (const [user, ids] of [['a', data.autoA], ['b', data.autoB]]) {
      const panel = el(`auto-${user}`); panel.replaceChildren();
      ids.forEach(id => { const song = byId.get(Number(id)); if (song) panel.append(text('div', 'auto-song', `${song.track_name} — ${song.artists} · ${song.track_genre}`)); });
    }
  }
  function setMode(next) {
    mode = next;
    document.querySelectorAll('.mode').forEach(button => button.classList.toggle('active', button.dataset.mode === mode));
    el('manual-controls').classList.toggle('hidden', mode !== 'MANUAL');
    el('automatic-preview').classList.toggle('hidden', mode !== 'AUTOMATIC');
    el('mode-description').textContent = mode === 'MANUAL'
      ? `Choose 5–${data.maxSongs} different songs per person. These exact selected songs become the input to the Linear Algebra model.`
      : 'Deterministic synthetic preference equations select 15 songs per profile; the exact songs shown become the input to the same Linear Algebra model.';
    updateWarning();
  }
  document.querySelectorAll('.mode').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
  for (const user of ['a', 'b']) {
    el(`search-${user}`).addEventListener('input', () => renderUser(user));
    el(`genre-${user}`).addEventListener('change', () => renderUser(user));
    renderUser(user);
  }
  showAutomatic(); updateWarning();

  const pretty = value => JSON.stringify(value, null, 2);
  function stage(number, title, concept, purpose, outcome) {
    const box = text('section', 'stage', '');
    box.append(text('span', 'stage-number', String(number).padStart(2, '0')));
    const body = text('div', 'stage-content', '');
    body.append(text('h4', '', title), text('p', '', `Concept · ${concept}`), text('p', '', `Purpose · ${purpose}`));
    if (outcome !== undefined) body.append(text('pre', '', pretty(outcome)));
    box.append(body); return box;
  }
  function addLegend(target, entries) {
    const legend = text('div', 'chart-legend', '');
    entries.forEach(([name, color, marker]) => { const item = text('span', '', ''); const swatch = text('i', `legend-marker ${marker || 'circle'}`, ''); swatch.style.setProperty('--swatch', color); item.append(swatch, document.createTextNode(name)); legend.append(item); });
    target.append(legend);
  }
  function axisText(parent, x, y, content, anchor = 'middle', className = 'axis-label') {
    const node = svg('text', {x, y, 'text-anchor': anchor, class: className}); node.textContent = content; parent.append(node); return node;
  }
  function makeChartSvg(container, width, height, labelText) {
    container.replaceChildren(); const root = svg('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': labelText, preserveAspectRatio: 'xMidYMid meet'}); container.append(root); return root;
  }
  function varianceChart(result) {
    const container = el('variance-chart');
    const width = 820, height = 300, left = 54, top = 25, plotH = 205, plotW = 730;
    const chart = makeChartSvg(container, width, height, 'PCA explained variance grouped chart');
    [0, .25, .5, .75, 1].forEach(v => { const y = top + plotH * (1-v); chart.append(svg('line',{x1:left,y1:y,x2:left+plotW,y2:y,class:'grid-line'})); axisText(chart,left-12,y+4,`${Math.round(v*100)}%`,'end'); });
    const valuesA = result.pca_explained_variance.user_a, valuesB = result.pca_explained_variance.user_b;
    const groupW = plotW / 3, bw = 54;
    for (let i=0;i<3;i++) {
      const center=left+groupW*(i+.5);
      [[valuesA[i],'bar-a',center-bw-5],[valuesB[i],'bar-b',center+5]].forEach(([value,cls,x])=>{
        const v=Math.max(0,Number(value)||0), h=plotH*Math.min(1,v), y=top+plotH-h;
        chart.append(svg('rect',{x,y,width:bw,height:h,rx:6,class:cls}));
        axisText(chart,x+bw/2,Math.max(15,y-7),`${(v*100).toFixed(1)}%`,'middle','bar-value');
      });
      axisText(chart,center,top+plotH+24,`PC${i+1}`);
    }
    addLegend(container,[['User A','#8b5cf6','circle'],['User B','#c084fc','circle']]);
  }
  function comparisonChart(result) {
    const container=el('baseline-chart'),width=820,height=280,left=70,top=24,plotH=190,plotW=690;
    const chart=makeChartSvg(container,width,height,'Average-vector cosine baseline compared with principal-angle compatibility, on a 0 to 100 percent scale');
    [0,25,50,75,100].forEach(value=>{const y=top+plotH*(1-value/100);chart.append(svg('line',{x1:left,y1:y,x2:left+plotW,y2:y,class:'grid-line'}));axisText(chart,left-12,y+4,`${value}%`,'end');});
    const bars=[['Average-vector cosine',Number(result.baseline_percent),'bar-a'],['Principal-angle compatibility',Number(result.compatibility),'bar-b']];
    bars.forEach(([name,value,cls],i)=>{const x=left+plotW*(i===0?.25:.65),w=120,h=plotH*Math.max(0,Math.min(100,value))/100,y=top+plotH-h;chart.append(svg('rect',{x,y,width:w,height:h,rx:7,class:cls}));axisText(chart,x+w/2,Math.max(15,y-8),`${value.toFixed(1)}%`,'middle','bar-value');axisText(chart,x+w/2,top+plotH+22,name);});
    axisText(chart,15,top+plotH/2,'Compatibility (%)','middle');
  }
  function principalAngleChart(angles) {
    const container=el('principal-angle-chart'),width=820,height=300,left=65,top=25,plotH=205,plotW=700;
    const chart=makeChartSvg(container,width,height,'Principal angles for directions 1, 2 and 3 in degrees, from 0 to 90');
    [0,22.5,45,67.5,90].forEach(value=>{const y=top+plotH*(1-value/90);chart.append(svg('line',{x1:left,y1:y,x2:left+plotW,y2:y,class:'grid-line'}));axisText(chart,left-12,y+4,`${value}°`,'end');});
    angles.forEach((angle,i)=>{const value=Math.max(0,Math.min(90,Number(angle))),x=left+plotW*((i+.5)/3)-44,h=plotH*value/90,y=top+plotH-h;chart.append(svg('rect',{x,y,width:88,height:h,rx:7,class:'bar-a'}));axisText(chart,x+44,Math.max(15,y-8),`${Number(angle).toFixed(2)}°`,'middle','bar-value');axisText(chart,x+44,top+plotH+22,`Direction ${i+1}`);});
    axisText(chart,15,top+plotH/2,'Angle (degrees)','middle');
  }
  function scatterChart(containerId, pca, includePlaylist) {
    const container=el(containerId), width=860,height=430,left=66,right=30,top=24,bottom=52;
    const chart=makeChartSvg(container,width,height,container.getAttribute('aria-label'));
    let series=[['User A',pca.user_a,'dot-a'],['User B',pca.user_b,'dot-b']];
    if(includePlaylist) series.push(['Blend recommendations',pca.playlist,'dot-rec']);
    const all=series.flatMap(s=>s[1]);
    const xs=all.map(d=>Number(d.PC1)),ys=all.map(d=>Number(d.PC2));
    const bound=(values)=>{const lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.08,.25);return [lo-pad,hi+pad];};
    const [x0,x1]=bound(xs),[y0,y1]=bound(ys),plotW=width-left-right,plotH=height-top-bottom;
    const xMap=x=>left+(x-x0)/(x1-x0)*plotW,yMap=y=>top+plotH-(y-y0)/(y1-y0)*plotH;
    for(let i=0;i<=4;i++){
      const x=left+plotW*i/4,y=top+plotH*i/4;
      chart.append(svg('line',{x1:x,y1:top,x2:x,y2:top+plotH,class:'grid-line'}),svg('line',{x1:left,y1:y,x2:left+plotW,y2:y,class:'grid-line'}));
      axisText(chart,x,top+plotH+20,(x0+(x1-x0)*i/4).toFixed(1));
      axisText(chart,left-11,y+4,(y1-(y1-y0)*i/4).toFixed(1),'end');
    }
    chart.append(svg('line',{x1:left,y1:top+plotH,x2:left+plotW,y2:top+plotH,class:'axis-line'}),svg('line',{x1:left,y1:top,x2:left,y2:top+plotH,class:'axis-line'}));
    axisText(chart,left+plotW/2,height-9,'PC1');
    const yLabel=axisText(chart,15,top+plotH/2,'PC2','middle'); yLabel.setAttribute('transform',`rotate(-90 15 ${top+plotH/2})`);
    series.forEach(([name,records,cls])=>records.forEach(record=>{
      const x=xMap(Number(record.PC1)),y=yMap(Number(record.PC2)); let point;
      if(cls==='dot-b') point=svg('rect',{x:x-4.3,y:y-4.3,width:8.6,height:8.6,rx:1,class:cls});
      else if(cls==='dot-rec') point=svg('path',{d:`M ${x} ${y-7} L ${x+7} ${y} L ${x} ${y+7} L ${x-7} ${y} Z`,class:cls});
      else point=svg('circle',{cx:x,cy:y,r:5,class:cls});
      const pointLabel=`${name}: ${record.track_name} — ${record.artists}; PC1 ${Number(record.PC1).toFixed(2)}, PC2 ${Number(record.PC2).toFixed(2)}`;
      point.setAttribute('tabindex','0');point.setAttribute('aria-label',pointLabel);const title=svg('title');title.textContent=pointLabel;point.append(title);chart.append(point);
    }));
    addLegend(container,series.map(([name,,cls])=>[name,cls==='dot-a'?'#8b5cf6':cls==='dot-b'?'#c084fc':'#f97316',cls==='dot-b'?'square':cls==='dot-rec'?'diamond':'circle']));
  }
  function featureChart(comparison) {
    const container=el('feature-chart'), width=860,height=390,left=185,right=55,top=23,rowH=47,plotW=width-left-right;
    const chart=makeChartSvg(container,width,height,'Display-normalized selected collection feature comparison');
    const values=[...comparison.display_mean_a,...comparison.display_mean_b].map(Number),limit=Math.max(1.5,Math.ceil(Math.max(...values.map(Math.abs))*2)/2),zero=left+plotW/2;
    [-limit,-limit/2,0,limit/2,limit].forEach(v=>{const x=zero+v/limit*(plotW/2);chart.append(svg('line',{x1:x,y1:top-4,x2:x,y2:top+rowH*7-5,class:v===0?'axis-line':'grid-line'}));axisText(chart,x,top+rowH*7+15,v.toFixed(1));});
    comparison.labels.forEach((feature,i)=>{
      const y=top+i*rowH; axisText(chart,left-12,y+17,feature.replaceAll('_',' '),'end','feature-label');
      const a=Number(comparison.display_mean_a[i]),b=Number(comparison.display_mean_b[i]);
      [[a,'bar-a',y],[b,'bar-b',y+17]].forEach(([value,cls,barY])=>{const w=Math.abs(value)/limit*(plotW/2),x=value>=0?zero:zero-w;chart.append(svg('rect',{x,y:barY,width:w,height:10,rx:4,class:cls}));});
    });
    addLegend(container,[['User A mean','#8b5cf6','circle'],['User B mean','#c084fc','circle']]);
    container.append(text('p','chart-footnote',comparison.display_label));
  }
  function renderPlaylist(songs) {
    const body=el('playlist');body.replaceChildren();
    songs.forEach(song=>{
      const row=document.createElement('tr');
      const rank=text('td','rank-cell',String(song.rank));
      const track=text('td','track-cell',song.track_name);track.append(text('small','',`ID ${song.track_id}`));
      row.append(rank,track,text('td','',song.artists),text('td','genre-cell',song.track_genre),text('td','score-cell',Number(song.blend_score).toFixed(3)),text('td','',Number(song.fit_to_A).toFixed(3)),text('td','',Number(song.fit_to_B).toFixed(3)),text('td','',Number(song.shared_fit).toFixed(3)),text('td','',Number(song.projection_error).toFixed(3)));
      body.append(row);
    });
  }
  function supportUser(targetId, info) {
    const rows=[['Matrix',info.matrix_shape.join(' × ')],['Songs',info.count],['Rank',info.rank],['Nullity',info.nullity],['Independent directions',info.rank],['Taste dimension',3],['RREF pivot columns',info.pivot_columns.join(', ') || 'none']];
    const list=text('dl','support-stats','');rows.forEach(([key,value])=>{list.append(text('dt','',key),text('dd','',String(value)));});el(targetId).replaceChildren(list);
  }
  function renderPipeline(result) {
    const pipeline=el('pipeline');pipeline.replaceChildren();
    const A=result.user_a,B=result.user_b;
    pipeline.append(stage(1,'Dataset & standardization','One global StandardScaler transforms all seven columns of the full dataset.','Keep BPM and 0–1 measurements in comparable coordinates.',{dataset_songs:data.songs.length,features:result.feature_order,scaler_scope:'complete dataset'}));
    pipeline.append(stage(2,'Matrix representation','Rows of X are songs; the seven features form columns. A and B put songs into columns.','Turn each exact selection into its 7 × n input matrix.',{X:[data.songs.length,7],A:A.matrix_shape,B:B.matrix_shape,mode:result.mode}));
    const pair=(field)=>({user_a:A[field],user_b:B[field]});
    pipeline.append(stage(3,'RREF','Gauss–Jordan elimination with pivoting reduces each selected matrix.','Expose pivot columns and independent song directions.',pair('rref')));
    pipeline.append(stage(4,'Rank / nullity / independence','Rank counts independent columns; nullity = columns − rank.','Describe each full selected-song span; nullity is not overlap between users.',{user_a:{rank:A.rank,nullity:A.nullity,columns:A.count,rank_plus_nullity:A.rank+A.nullity,pivot_columns_1_based:A.pivot_columns},user_b:{rank:B.rank,nullity:B.nullity,columns:B.count,rank_plus_nullity:B.rank+B.nullity,pivot_columns_1_based:B.pivot_columns}}));
    pipeline.append(stage(5,'Independent basis','RREF pivot columns select an independent spanning set.','Represent both full spans without redundant song vectors.',{user_a:A.independent_song_basis,user_b:B.independent_song_basis}));
    pipeline.append(stage(6,'Gram–Schmidt','Modified Gram–Schmidt orthogonalizes all independent basis columns; the first 3 form each modeled taste space.','Provide full orthonormal spans and the 3D Q matrices used for principal angles.',{user_a:{full_orthonormal_span:A.full_orthonormal_span,taste_basis_Q:A.taste_basis,Q_transpose_Q:A.orthogonality_check},user_b:{full_orthonormal_span:B.full_orthonormal_span,taste_basis_Q:B.taste_basis,Q_transpose_Q:B.orthogonality_check}}));
    pipeline.append(stage(7,'Projection','P = QQᵀ and x̂ = QQᵀx.','Find the closest point in each taste subspace to the selected collection mean.',{user_a:{mean:A.mean_vector,projection:A.projection},user_b:{mean:B.mean_vector,projection:B.projection}}));
    pipeline.append(stage(8,'Least squares','For orthonormal Q, c = Qᵀx minimizes ||Qc − x||².','Give the best linear combination and residual for each collection.',{user_a:{coefficients:A.least_squares_coefficients,residual:A.projection_residual,residual_norm:A.projection_residual_norm},user_b:{coefficients:B.least_squares_coefficients,residual:B.projection_residual,residual_norm:B.projection_residual_norm}}));
    pipeline.append(stage(9,'Covariance','C = X_cᵀX_c/(n−1), where rows are selected songs centered by the sample mean.','Measure feature variation within both collections.',pair('covariance')));
    pipeline.append(stage(10,'Eigenvalues / eigenvectors','A symmetric covariance matrix is decomposed using eigh and sorted by descending eigenvalue.','Identify feature-space variation directions and magnitudes for both users.',{user_a:{eigenvalues:A.eigenvalues,eigenvectors:A.eigenvectors},user_b:{eigenvalues:B.eigenvalues,eigenvectors:B.eigenvectors}}));
    pipeline.append(stage(11,'Diagonalization','C = VΛVᵀ.','Check covariance reconstruction from eigenvectors and eigenvalues.',{user_a:A.diagonalization_reconstruction_error,user_b:B.diagonalization_reconstruction_error}));
    pipeline.append(stage(12,'PCA','Leading covariance eigenvectors explain the greatest variation.','Show explained variance; PCA remains separate from the Gram–Schmidt taste basis.',{user_a:A.pca_explained_variance_ratio,user_b:B.pca_explained_variance_ratio}));
    pipeline.append(stage(13,'SVD','M = Q_AᵀQ_B = UΣVᵀ.','Compare the two 3D orthonormal taste bases.',{cross_basis_matrix:result.cross_basis_matrix,singular_values:result.singular_values}));
    pipeline.append(stage(14,'Principal angles','σᵢ = cos(θᵢ), θᵢ = arccos(σᵢ).','Translate singular values into direction-by-direction angles.',result.principal_angle_table));
    pipeline.append(stage(15,'Shared taste directions','Corresponding principal vectors are sign-aligned, averaged, then orthonormalized.','Expose directions shared by the modeled taste spaces.',{orthonormal_shared_directions:result.shared_directions,principal_vector_pairs:result.shared_direction_details}));
    pipeline.append(stage(16,'Playlist scoring','blend = 0.4 fit_A + 0.4 fit_B + 0.2 shared_fit.','Rank dataset songs using the same global scaler and omit all selected IDs.',{candidate_pool:'full dataset excluding selected songs',scoring:'0.4 fit_A + 0.4 fit_B + 0.2 shared_fit'}));
    pipeline.append(stage(17,'Final playlist','Each row is a real dataset track with its component fits and shared-space residual.','Present the ten highest ranked unselected tracks.',result.playlist));
  }
  function render(result) {
    el('results').classList.remove('hidden');el('result-content').classList.remove('hidden');el('api-error').classList.add('hidden');
    el('compatibility').textContent=`${result.compatibility.toFixed(1)}%`;
    const stats=el('result-stats');stats.replaceChildren();
    [[`User A songs`,result.user_a.count],[`User B songs`,result.user_b.count],[`User A rank`,result.user_a.rank],[`User B rank`,result.user_b.rank],[`Taste dimension`,result.user_a.taste_basis[0].length]].forEach(([name,value])=>{const item=text('span','result-stat','');item.append(text('b','',String(value)),text('small','',name));stats.append(item);});
    el('compatibility-interpretation').textContent=result.compatibility_interpretation;
    el('mean-angle').textContent=`${Number(result.mean_principal_angle).toFixed(1)}°`;
    el('baseline-cosine').textContent=Number(result.baseline_cosine).toFixed(3);
    el('baseline-percent').textContent=`${Number(result.baseline_percent).toFixed(1)}% remapped display score`;
    el('baseline-subspace').textContent=`${result.compatibility.toFixed(1)}%`;
    comparisonChart(result);principalAngleChart(result.principal_angles);
    el('angles').replaceChildren(...result.principal_angles.map((angle,i)=>{const item=text('div','angle',`${Number(angle).toFixed(1)}°`);item.prepend(text('b','',`Direction ${i+1}`));return item;}));
    const angleBody=el('angle-table');angleBody.replaceChildren();
    result.principal_angle_table.forEach(entry=>{const row=document.createElement('tr');row.append(text('td','',String(entry.direction)),text('td','numeric',Number(entry.singular_value).toFixed(4)),text('td','angle-value',`${Number(entry.angle_degrees).toFixed(2)}°`),text('td','',Number(entry.overlap_strength).toFixed(4)));angleBody.append(row);});
    supportUser('support-a',result.user_a);supportUser('support-b',result.user_b);
    const orthErrorA=Math.sqrt(result.user_a.orthogonality_check.flat().reduce((sum,value,i)=>{const row=Math.floor(i/3),col=i%3;return sum+(value-(row===col?1:0))**2;},0));
    const orthErrorB=Math.sqrt(result.user_b.orthogonality_check.flat().reduce((sum,value,i)=>{const row=Math.floor(i/3),col=i%3;return sum+(value-(row===col?1:0))**2;},0));
    el('orth-a').textContent=`error ${orthErrorA.toExponential(1)}`;el('orth-b').textContent=`error ${orthErrorB.toExponential(1)}`;
    const shared=el('shared-directions');shared.replaceChildren();
    result.shared_direction_details.forEach(direction=>{const row=text('div','shared-direction','');row.append(text('b','',`Direction ${direction.direction}`),text('span','',`${direction.angle_degrees.toFixed(1)}° angle`),text('span','',`${direction.overlap_strength.toFixed(3)} overlap strength`));shared.append(row);});
    varianceChart(result);scatterChart('taste-pca-chart',result.pca_visualization,false);scatterChart('recommendation-pca-chart',result.pca_visualization,true);featureChart(result.feature_comparison);renderPlaylist(result.playlist);renderPipeline(result);
    el('results').scrollIntoView({behavior:'smooth',block:'start'});
  }
  el('back-to-songs').addEventListener('click',()=>{
    el('results').classList.add('hidden');
    el('select-taste').scrollIntoView({behavior:'smooth',block:'start'});
  });
  el('analyze-button').addEventListener('click',async()=>{
    const error=el('api-error');error.classList.add('hidden');el('results').classList.remove('hidden');el('result-content').classList.add('hidden');
    if(mode==='MANUAL'&&(selected.a.size<5||selected.b.size<5)){showWarning('Select at least 5 songs for each user before analysis.');return;}
    const button=el('analyze-button');button.disabled=true;button.textContent='Calculating linear algebra…';
    try{
      const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,user_a:[...selected.a],user_b:[...selected.b]})});
      const result=await response.json();if(!response.ok)throw new Error(result.error||'Analysis failed.');render(result);
    }catch(exception){error.textContent=exception.message;error.classList.remove('hidden');}
    finally{button.disabled=false;button.innerHTML='Analyze taste <span>→</span>';}
  });
})();
