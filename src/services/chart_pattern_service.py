import logging
from typing import List, Dict, Any

from src.models.equity import DetectedPattern, PatternPoint, Trendline

logger = logging.getLogger(__name__)

class ChartPatternService:
    def _find_pivot_points(
        self,
        historical_data: List[Dict[str, Any]],
        span: int = 5
    ) -> List[Dict[str, Any]]:
        pivots = []
        if len(historical_data) <= 2 * span:
            return []

        data_with_index = [{'index': i, **d} for i, d in enumerate(historical_data)]

        for i in range(span, len(data_with_index) - span):
            current_high = data_with_index[i].get('high')
            current_low = data_with_index[i].get('low')
            if current_high is None or current_low is None:
                continue

            window = data_with_index[i - span : i + span + 1]
            is_pivot_high = all(current_high >= p.get('high', current_high) for p in window)
            is_pivot_low = all(current_low <= p.get('low', current_low) for p in window)

            if is_pivot_high:
                pivots.append({'type': 'high', 'index': i, 'time': historical_data[i]['date'], 'value': current_high})
            if is_pivot_low:
                pivots.append({'type': 'low', 'index': i, 'time': historical_data[i]['date'], 'value': current_low})
        
        if not pivots:
            return []

        pivots.sort(key=lambda p: p['index'])
        
        unique_pivots = [pivots[0]]
        for i in range(1, len(pivots)):
            if pivots[i]['type'] != unique_pivots[-1]['type']:
                unique_pivots.append(pivots[i])
            else:
                if pivots[i]['type'] == 'high':
                    if pivots[i]['value'] > unique_pivots[-1]['value']:
                        unique_pivots[-1] = pivots[i]
                else:
                    if pivots[i]['value'] < unique_pivots[-1]['value']:
                        unique_pivots[-1] = pivots[i]
        return unique_pivots

    def _detect_double_top_bottom(self, pivots: List[Dict[str, Any]]) -> List[DetectedPattern]:
        patterns = []
        if len(pivots) < 3: return []
        for i in range(len(pivots) - 2):
            p1, p2, p3 = pivots[i:i+3]
            if p1['value'] == 0 or p2['value'] == 0: continue

            depth = abs(p2['value'] - (p1['value'] + p3['value']) / 2) / p2['value']
            if depth < 0.02: continue

            is_double_top = p1['type'] == 'high' and p2['type'] == 'low' and p3['type'] == 'high'
            if is_double_top and abs(p1['value'] - p3['value']) / p1['value'] < 0.03:
                points = [PatternPoint(time=p['time'], value=p['value']) for p in [p1, p2, p3]]
                patterns.append(DetectedPattern(pattern_name="Double Top", path_points=points))

            is_double_bottom = p1['type'] == 'low' and p2['type'] == 'high' and p3['type'] == 'low'
            if is_double_bottom and abs(p1['value'] - p3['value']) / p1['value'] < 0.03:
                points = [PatternPoint(time=p['time'], value=p['value']) for p in [p1, p2, p3]]
                patterns.append(DetectedPattern(pattern_name="Double Bottom", path_points=points))
        return patterns

    def _detect_head_and_shoulders(self, pivots: List[Dict[str, Any]]) -> List[DetectedPattern]:
        patterns = []
        if len(pivots) < 5: return []
        for i in range(len(pivots) - 4):
            p1, p2, p3, p4, p5 = pivots[i:i+5]
            is_inv_hs = (p1['type'] == 'low' and p2['type'] == 'high' and 
                         p3['type'] == 'low' and p4['type'] == 'high' and p5['type'] == 'low')
            if is_inv_hs:
                left_shoulder, head, right_shoulder = p1, p3, p5
                if head['value'] < left_shoulder['value'] and head['value'] < right_shoulder['value']:
                    if left_shoulder['value'] != 0 and abs(left_shoulder['value'] - right_shoulder['value']) / left_shoulder['value'] < 0.05:
                        points = [PatternPoint(time=p['time'], value=p['value']) for p in [p1, p2, p3, p4, p5]]
                        patterns.append(DetectedPattern(pattern_name="Inverted Head and Shoulders", path_points=points))
        return patterns
        
    def _detect_triangles(self, pivots: List[Dict[str, Any]]) -> List[DetectedPattern]:
        patterns = []
        if len(pivots) < 5: return []
        for i in range(len(pivots) - 4):
            window = pivots[i:i+5]
            highs = sorted([p for p in window if p['type'] == 'high'], key=lambda p: p['index'])
            lows = sorted([p for p in window if p['type'] == 'low'], key=lambda p: p['index'])

            if len(highs) < 2 or len(lows) < 2: continue
            
            h1, h2 = highs[0], highs[-1]
            l1, l2 = lows[0], lows[-1]
            
            if h1['value'] == 0 or l1['value'] == 0: continue
            
            tolerance = 0.015
            is_ascending_lows = l2['value'] > l1['value']
            is_descending_highs = h2['value'] < h1['value']
            are_highs_flat = abs(h1['value'] - h2['value']) / h1['value'] < tolerance
            are_lows_flat = abs(l1['value'] - l2['value']) / l1['value'] < tolerance
            
            upper_trendline = Trendline(line_type="resistance", points=[PatternPoint(time=h1['time'], value=h1['value']), PatternPoint(time=h2['time'], value=h2['value'])])
            lower_trendline = Trendline(line_type="support", points=[PatternPoint(time=l1['time'], value=l1['value']), PatternPoint(time=l2['time'], value=l2['value'])])
            
            if are_highs_flat and is_ascending_lows:
                patterns.append(DetectedPattern(pattern_name="Ascending Triangle", trendlines=[upper_trendline, lower_trendline]))
            elif are_lows_flat and is_descending_highs:
                patterns.append(DetectedPattern(pattern_name="Descending Triangle", trendlines=[upper_trendline, lower_trendline]))
            elif is_descending_highs and is_ascending_lows:
                patterns.append(DetectedPattern(pattern_name="Symmetrical Triangle", trendlines=[upper_trendline, lower_trendline]))
        return patterns

    def _filter_overlapping_patterns(self, patterns: List[DetectedPattern]) -> List[DetectedPattern]:
        if not patterns: return []
        
        def get_start_time(p: DetectedPattern):
            all_points = p.path_points or []
            if p.trendlines:
                for line in p.trendlines:
                    all_points.extend(line.points)
            return min(point.time for point in all_points) if all_points else ""

        patterns.sort(key=get_start_time)
        
        filtered_list = []
        last_pattern_end_time = ""

        for pattern in patterns:
            all_points = pattern.path_points or []
            if pattern.trendlines:
                for line in pattern.trendlines:
                    all_points.extend(line.points)
            if not all_points: continue
            
            pattern_start_time = min(p.time for p in all_points)
            if pattern_start_time >= last_pattern_end_time:
                filtered_list.append(pattern)
                last_pattern_end_time = max(p.time for p in all_points)
        return filtered_list

    def find_all_patterns(self, historical_data: List[Dict[str, Any]]) -> List[DetectedPattern]:
        if len(historical_data) < 30:
            return []

        pivots = self._find_pivot_points(historical_data)
        if len(pivots) < 4:
            return []
            
        potential_patterns = []
        potential_patterns.extend(self._detect_triangles(pivots))
        potential_patterns.extend(self._detect_head_and_shoulders(pivots))
        
        if not potential_patterns:
            potential_patterns.extend(self._detect_double_top_bottom(pivots))
        
        if not potential_patterns:
            return []

        final_patterns = self._filter_overlapping_patterns(potential_patterns)
        
        if final_patterns:
            pattern_names = [p.pattern_name for p in final_patterns]
            logger.info(f"Found {len(final_patterns)} distinct, significant patterns: {pattern_names}")
            
        return final_patterns

chart_pattern_service = ChartPatternService()